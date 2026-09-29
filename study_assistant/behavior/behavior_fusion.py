"""行为融合引擎：把特征变成"他现在在做什么"的概率分布。

这个模块**只做瞬时打分**，不做去抖、不做滞后 —— 那是
`temporal_engine.py` 和 `state_machine.py` 的职责。分离的好处是打分
逻辑可以单独用录制数据回放调试。

打分原则（非常重要）：
  * 每个行为都有自己的**证据链**，缺一环就大幅掉分，而不是靠单点
    阈值。比如写字需要"手在书写区" + "有持续的小幅动作" + "幅度不
    夸张"三者同时成立；
  * 手在动**不等于**分心。写字的手一直在动。真正的判据是"动作的
    幅度/方向"与"动作落在哪个区域"；
  * 手机是唯一一个"见到就基本能定罪"的证据（因为它对学习场景没有
    正当理由），但仍需持续时长过滤，避免一闪而过；
  * **ROI 是"先验"，不是"证据"**。ROI 只知道"这个位置通常是键盘"，
    不知道那里到底有没有键盘。所以位置类信号只能做**召回兜底**，且
    必须再过一道**正向动作证据**才能生效 —— 否则手只要落在框里就能
    定罪，物体检测形同虚设（曾经就是这样：`keyboard_contact` 里的
    物体分支被位置分支用 `or` + `max()` 完全吞掉，一次都没生效过）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .states import Behavior


def _ramp(x: float, lo: float, hi: float) -> float:
    """x 从 lo 线性升到 hi，返回 0..1。"""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _fall(x: float, lo: float, hi: float) -> float:
    """x 从 lo 线性降到 hi，返回 1..0。"""
    return 1.0 - _ramp(x, lo, hi)


#: 打字签名的四个归一化项名（顺序固定，展示与测试都按它来）
TYPING_TERM_NAMES = ("方向反转", "脉冲波动", "幅度收敛", "有实质动作")

#: 打字签名的四个原始读数名
TYPING_RAW_NAMES = ("dir_rate", "std_ratio", "range_3s", "energy")

#: 打分**平分**时的裁决顺序（靠前者赢）：净学习 > 不确定 > 分心。
#:
#: 为什么需要它：`max()` 默认返回"先遇到的那个"，所以不显式定义顺序时，
#: 谁能赢取决于本文件里语句的书写顺序 —— 一条隐式规则，重构一次就会变。
#: 实测踩过：阅读与玩手机同为 1.000 时，因为手机那一段写在前面，**看书的人
#: 被判成了玩手机**。
#:
#: 分心类排在最后同样是刻意取舍：平分本身就说明证据不足，此时不该给用户
#: 记一笔分心（假分心的代价是虚假提醒 + 专注分被拉低）。
_TIE_BREAK_ORDER = (
    Behavior.READING,
    Behavior.WRITING,
    Behavior.COMPUTER_STUDY,
    Behavior.IDLE,
    Behavior.HAND_AWAY,
    Behavior.AWAY,
    Behavior.UNKNOWN,
    Behavior.FIDGETING,
    Behavior.PHONE_USE,
)


def _tie_rank(behavior) -> int:
    try:
        return _TIE_BREAK_ORDER.index(behavior)
    except ValueError:
        return len(_TIE_BREAK_ORDER)


def typing_signature_from_raw(
    raw: dict,
    *,
    dir_lo: float,
    dir_hi: float,
    burst_lo: float,
    burst_hi: float,
    range_max: float,
    energy_lo: float,
    energy_hi: float,
) -> dict:
    """打字签名的**纯函数内核**：原始读数 + 阈值 → 四项与几何平均分。

    单独抽出来是为了让「真机标定」能拿**候选阈值**重算 —— 标定工具必须
    能用和线上同一份公式验证"建议的阈值到底让不让我的打字过"。如果这段
    公式被复制到工具里，两边迟早会走偏，标出来的阈值就是废的。

    四项缺任何一环 → 乘积为 0 → 总分 0。这是刻意的：证据链缺一环就该
    大幅掉分，而不是靠另外三项加权补回来。
    """
    terms = {
        "方向反转": _ramp(raw["dir_rate"], dir_lo, dir_hi),
        "脉冲波动": _ramp(raw["std_ratio"], burst_lo, burst_hi),
        "幅度收敛": _fall(raw["range_3s"], range_max, range_max * 2.0),
        "有实质动作": _ramp(raw["energy"], energy_lo, energy_hi),
    }

    product = 1.0
    for value in terms.values():
        product *= max(0.0, value)

    return {
        "terms": terms,
        "score": product ** 0.25 if product > 0 else 0.0,
    }


@dataclass
class BehaviorVerdict:
    """一帧的打分结果。"""

    timestamp: float = 0.0
    scores: dict = field(default_factory=dict)      # {Behavior: 0..1} 归一化
    raw_scores: dict = field(default_factory=dict)  # {Behavior: 0..1} 原始分
    evidence: dict = field(default_factory=dict)    # {Behavior: [str, ...]}
    top: Behavior = Behavior.UNKNOWN
    confidence: float = 0.0

    def score(self, behavior: Behavior) -> float:
        return float(self.scores.get(behavior, 0.0))

    def as_dict(self) -> dict:
        return {b.value: round(s, 3) for b, s in self.scores.items() if s > 0.01}


class BehaviorFusionEngine:
    """多行为并行打分。"""

    def __init__(self, config):
        b = config.behavior

        # --- 写字 ---
        writing = b.get("writing", {})
        self.write_energy_lo = float(writing.get("min_motion_energy", 0.015))
        self.write_energy_hi = float(writing.get("full_motion_energy", 0.05))
        self.write_range_max = float(writing.get("max_motion_range", 0.25))

        # --- 阅读 ---
        reading = b.get("reading", {})
        self.read_energy_static = float(reading.get("static_motion_energy", 0.006))
        # 静止度开始衰减的能量上限。**这个值原来是 0.018，离下限 0.006 太近**：
        # 意味着能量一到 0.018 静止度就归零，而"写字"的能量项要到 0.05 才满分 ——
        # 中间 0.018~0.05 这一大片区域里，阅读分已经死在硬下限、写字分还在往上爬，
        # 就出现"写字和阅读分不开"。放宽到 0.06，让静止度能平滑地覆盖
        # "认真读书（≈0.002）→ 轻微翻动（≈0.02）→ 写字（≥0.05）"整条轴。
        self.read_energy_max = float(reading.get("max_motion_energy", 0.06))
        # 静止是阅读的**必要条件**：手在动就不可能是在看书。
        # `static_energy_grace` 就是这条否决线 —— 超过它，阅读分直接归零，
        # 而不是只把 0.45 的权重扣掉一部分。
        self.read_static_grace = float(reading.get("static_energy_grace", 0.018))

        # --- 手机 ---
        phone = b.get("phone", {})
        self.phone_candidate = float(phone.get("candidate_seconds", 2.0))
        # 「随手看一眼」的时长上限：超过它才算真的在玩，而不是瞥一眼时间
        self.phone_quick_look = float(phone.get("quick_look_seconds", 5.0))
        # **默认要求「检出手机物体」才定罪**：位置（phone ROI）单独不作数。
        # 假的"玩手机"会记分心、触发提醒、拉低专注分，代价远高于漏判；
        # 而"手落在 phone 框里"根本不说明那里有手机（看书/敲键盘都能落进去）。
        # 置 false 可恢复"位置也算"的旧行为，但会整条折价（宁漏不误的取舍）。
        self.phone_require_object = bool(phone.get("require_object_contact", True))
        self.phone_position_only_penalty = float(
            phone.get("position_only_penalty", 0.6)
        )

        # --- 小动作 ---
        fidget = b.get("fidgeting", {})
        self.fid_range_lo = float(fidget.get("min_motion_range", 0.30))
        self.fid_range_hi = float(fidget.get("full_motion_range", 0.55))
        self.fid_dir_min = float(fidget.get("min_direction_change", 0.5))
        # 小动作必须先**真的有动作**：手几乎不动时，关键点的微小抖动也能
        # 刷出很高的方向反转率（实测 4/s），没有能量下限就会把"手压着
        # 键盘发呆"和"手搭在桌上不动"判成小动作。
        self.fid_energy_min = float(fidget.get("min_motion_energy", 0.02))

        # --- 电脑学习 / 打字动作签名 ---
        # 「手落在键盘区」是**位置**，「正在打字」是**动作**。位置不能单独
        # 定罪，必须再过这四个量的几何平均（缺一环就把结果压到 0）。
        computer = b.get("computer", {})
        self.comp_typing_dir_lo = float(computer.get("typing_dir_min", 2.0))
        self.comp_typing_dir_hi = float(computer.get("typing_dir_full", 6.0))
        self.comp_typing_burst_lo = float(computer.get("typing_burst_min", 0.45))
        self.comp_typing_burst_hi = float(computer.get("typing_burst_full", 0.75))
        self.comp_typing_range_max = float(computer.get("typing_range_max", 0.12))
        self.comp_typing_energy_lo = float(computer.get("typing_energy_min", 0.10))
        self.comp_typing_energy_hi = float(computer.get("typing_energy_full", 0.25))
        self.comp_typing_min = float(computer.get("typing_min_score", 0.45))
        # 只靠位置、没检出键盘物体时整体折价 —— 让「物体检测」真的能改变结果
        self.comp_position_only_penalty = float(
            computer.get("position_only_penalty", 0.6)
        )
        # True = 完全忽略位置通路，纯检测驱动（需要摄像头能拍到键盘）
        self.comp_require_object = bool(computer.get("require_object_contact", False))
        # 打字签名成立时不再判小动作（键击是**有目的**的动作）
        self.comp_suppress_fidget = bool(computer.get("suppress_fidget", True))

        self.min_confidence = float(b.get("min_confidence", 0.45))

        # 组装行为判定用的行为集合
        self._score_keys = [
            Behavior.WRITING,
            Behavior.READING,
            Behavior.COMPUTER_STUDY,
            Behavior.PHONE_USE,
            Behavior.FIDGETING,
            Behavior.IDLE,
            Behavior.HAND_AWAY,
            Behavior.AWAY,
        ]

    # ------------------------------------------------------------------
    # 打字动作签名 —— **唯一实现**
    # ------------------------------------------------------------------

    def typing_signature(self, motions: dict) -> dict:
        """打字动作签名：四个原始读数 + 四个归一化项 + 几何平均分。

        「手落在键盘区」只是**位置**，它只知道"这个位置通常是键盘"，不知道
        那里到底有没有键盘、手在不在敲。真正的打字是**脉冲式**的：

          * 方向反转率极高：键击是离散的来回，一次按键就是一次方向翻转；
          * 速度波动比高（speed_std / speed）：击打—停顿交替，速度忽高忽低；
          * 幅度很小：手在键盘范围内，不会真的跑开；
          * 有实质运动：排除"手压着键盘发呆"（速度接近 0）。

        四项用**几何平均**合成：缺任何一环都会把整体压到 0，符合本模块
        「证据链缺一环就大幅掉分」的原则，而不是靠加权求和互相补偿。

        这个方法必须只有这一份实现：判定（`update`）与真机标定工具
        （`tools/calibrate_typing.py`）都调它。否则标定工具量出来的
        「签名分」和线上真正用的那套公式不是一回事，标出来的阈值就是废的。
        """
        m1 = motions.get(1.0, {}) or {}
        m3 = motions.get(3.0, {}) or {}

        # 只用样本足够的手（samples < 2 时运动统计没有意义）
        energies = [mo.speed for mo in m1.values() if mo.samples >= 2]
        dir_rates = [mo.direction_change_rate for mo in m1.values() if mo.samples >= 2]
        std_ratios = [
            (mo.speed_std / mo.speed) if mo.speed > 1e-6 else 0.0
            for mo in m1.values() if mo.samples >= 2
        ]
        ranges_3s = [mo.motion_range for mo in m3.values() if mo.samples >= 2]

        raw = {
            "dir_rate": max(dir_rates) if dir_rates else 0.0,
            "std_ratio": max(std_ratios) if std_ratios else 0.0,
            "range_3s": max(ranges_3s) if ranges_3s else 0.0,
            "energy": max(energies) if energies else 0.0,
        }

        result = typing_signature_from_raw(
            raw,
            dir_lo=self.comp_typing_dir_lo,
            dir_hi=self.comp_typing_dir_hi,
            burst_lo=self.comp_typing_burst_lo,
            burst_hi=self.comp_typing_burst_hi,
            range_max=self.comp_typing_range_max,
            energy_lo=self.comp_typing_energy_lo,
            energy_hi=self.comp_typing_energy_hi,
        )
        result["raw"] = raw
        return result

    # ------------------------------------------------------------------

    def update(
        self,
        hands: list,
        motions: dict,
        interactions,
        presence,
        now: float,
        hands_absent_for: float = 0.0,
        person_absent_for: float = 0.0,
        person_present: bool = False,
        camera_ok: bool = True,
    ) -> BehaviorVerdict:
        """
        hands             —— list[Hand]，本帧（或最近一次推理）的手
        motions           —— {window_seconds: {hand_label: HandMotion}}
        interactions      —— InteractionReport
        presence          —— ObjectPresenceTracker
        hands_absent_for  —— 手已经消失多久
        person_absent_for —— 人也消失了多久
        person_present    —— 人是否被**真实检出**且在新鲜窗口内（用于否决离席）
        """
        verdict = BehaviorVerdict(timestamp=now)
        scores: dict = {}
        evidence: dict = {}

        # ---------- 0) 预处理：把手部运动合并成"整体运动" ----------
        m1 = motions.get(1.0, {}) or {}
        m3 = motions.get(3.0, {}) or {}

        energies = [mo.speed for mo in m1.values() if mo.samples >= 2]
        ranges_1s = [mo.motion_range for mo in m1.values() if mo.samples >= 2]
        ranges_3s = [mo.motion_range for mo in m3.values() if mo.samples >= 2]
        dir_rates = [mo.direction_change_rate for mo in m1.values() if mo.samples >= 2]
        locality = [mo.is_local for mo in m1.values() if mo.samples >= 2]
        # 速度波动比（speed_std / speed）是区分打字与写字的关键量，但它**只**
        # 被打字签名用到 —— 那段计算挪进了 `typing_signature()`，避免同一套
        # 公式在两个地方各写一份、日后改一处漏一处。
        max_energy = max(energies) if energies else 0.0
        max_range_1s = max(ranges_1s) if ranges_1s else 0.0
        max_range_3s = max(ranges_3s) if ranges_3s else 0.0
        max_dir = max(dir_rates) if dir_rates else 0.0
        any_local = any(locality) if locality else False

        hand_count = len(hands)
        has_hands = hand_count > 0

        # ---------- 1) 离席 / 手离开 ----------
        # 关键区分：**手不在画面 ≠ 人离席**。
        # 摄像头只拍到上半身时，手在桌子下面永远拍不到。若把"手消失很久"
        # 直接判成离席，会凭空记一笔离席时长、拉低专注分，还会在事件流里
        # 写"离开座位"。所以：只要人还被真实检出，离席分数必须归零，改由
        # HAND_AWAY 去描述"看不到手"这个证据缺口。
        away_score = 0.0
        away_ev = []
        if not has_hands:
            if person_present:
                away_ev.append("人在画面内，排除离席")
            else:
                away_score = _ramp(hands_absent_for, 0.5, 2.5)
                away_ev.append(f"手消失 {hands_absent_for:.1f}s")
        if person_absent_for > 0 and not person_present:
            away_score = max(away_score, _ramp(person_absent_for, 1.0, 5.0))
            away_ev.append(f"人消失 {person_absent_for:.1f}s")
        scores[Behavior.AWAY] = away_score
        evidence[Behavior.AWAY] = away_ev

        hand_away_score = 0.0
        hand_away_ev = []
        if not has_hands:
            # 刚消失 → HAND_AWAY；消失更久且人也不在 → 分数交给 AWAY。
            # 但"人在画面里、只是手拍不到"是个**持续**状态：这时 HAND_AWAY
            # 不能衰减到 0，否则行为会退回"识别中"，用户就看不出问题其实
            # 出在摄像头角度 / 标定上，而不是自己没在学习。
            hand_away_score = _ramp(hands_absent_for, 0.2, 1.0)
            if person_present:
                hand_away_ev.append("人在画面内（手不在画面里）")
            else:
                hand_away_score *= _fall(hands_absent_for, 2.0, 5.0)
            hand_away_ev.append(f"手离开画面 {hands_absent_for:.1f}s")
        scores[Behavior.HAND_AWAY] = hand_away_score
        evidence[Behavior.HAND_AWAY] = hand_away_ev

        # ---------- 2) 手机 ----------
        # 判据链：**检出手机物体 → 手与它发生接触**。位置（phone ROI）只是
        # 先验，默认**不作数** —— 出厂 phone 框是 x∈[0.02,0.20], y∈[0.50,0.92]
        # 那一整条左下竖带，看书时搭着的手、敲键盘时下探的手都可能落进去，
        # 而"位置对"完全不能说明那里有手机。
        # 误判代价在这里特别高：假的"玩手机"会凭空记一笔分心、触发提醒、
        # 拉低专注分，用户看到的结论与事实相反。所以宁漏不误。
        phone_score = 0.0
        phone_ev = []

        obj_ok = bool(interactions.phone_object_contact)
        roi_ok = bool(interactions.phone_roi_contact)
        # 位置兜底只在显式关掉「必须检出物体」时才启用（默认关闭）
        phone_ok = obj_ok or (roi_ok and not self.phone_require_object)

        if has_hands and phone_ok:
            dur = (
                interactions.phone_object_duration if obj_ok
                else interactions.phone_duration
            )
            phone_score = _ramp(dur, 0.6, self.phone_candidate)

            if obj_ok:
                phone_ev.append(f"检出手机且手与它接触 {dur:.1f}s")
            else:
                phone_score *= self.phone_position_only_penalty
                phone_ev.append(f"仅位置命中手机框（未检出手机）{dur:.1f}s，已折价")

            # 真的握在手里（关键点大量重合）比"手在手机附近"更确凿
            best_overlap = max(
                (m.overlap_ratio for m in interactions.measures
                 if m.object_label == "phone"),
                default=0.0,
            )
            if best_overlap >= 0.25:
                phone_score = min(1.0, phone_score + 0.15)
                phone_ev.append(f"握持明显 (overlap {best_overlap:.0%})")

            # 超过「随手看一眼」的时长 → 证据更强
            if dur >= self.phone_quick_look:
                phone_score = min(1.0, phone_score + 0.10)
                phone_ev.append(f"已持续 {dur:.0f}s，超过随手一看的时长")

            if not obj_ok:
                phone_ev.append("未检出手机物体 → 不判为玩手机")
        elif has_hands and roi_ok:
            # 手在手机框里、但没检出手机：如实说明为什么不算，否则用户只会
            # 看到"什么都不判"，不知道是缺哪一项证据
            phone_ev.append(
                "手落在手机框内，但未检出手机物体 → 位置先验不足以定罪"
            )

        scores[Behavior.PHONE_USE] = phone_score
        evidence[Behavior.PHONE_USE] = phone_ev

        # ---------- 3) 写字 ----------
        write_score = 0.0
        write_ev = []
        if has_hands and interactions.paper_contact:
            dur = interactions.paper_duration
            duration_term = _ramp(dur, 0.5, 2.0)
            energy_term = _ramp(max_energy, self.write_energy_lo, self.write_energy_hi)
            # 幅度过大会被判为"挥手"，写字是收敛的小幅动作
            range_term = _fall(max_range_1s, self.write_range_max, self.write_range_max * 1.8)
            local_term = 1.0 if any_local else 0.6

            write_score = (
                0.35 * duration_term
                + 0.30 * energy_term
                + 0.20 * range_term
                + 0.15 * local_term
            )
            write_ev.append(f"手在书写区 {dur:.1f}s")
            write_ev.append(f"运动能量 {max_energy:.3f}")
            write_ev.append(f"1s 幅度 {max_range_1s:.3f}")
        scores[Behavior.WRITING] = write_score
        evidence[Behavior.WRITING] = write_ev

        # ---------- 4) 阅读 ----------
        # 判据链：**手在书写区/书在画面 → 手确实是静止的**。
        #
        # 旧版是 `read_score = 0.55*时长 + 0.45*静止度`，两个问题叠在一起：
        #
        #  ① 时长项与写字**共用同一根计时器**（都取 interactions.paper_duration），
        #     且 2.5s 后恒为 1.0 → 阅读分有一条 **0.550 的硬下限**。写字时手当然
        #     也压在纸上、纸区停留当然也够久，于是"正在写字"时阅读照样拿 0.550，
        #     写字封顶 1.000 —— 两者最大差距只有 0.45，而且**永远弥合不了**。
        #  ② 静止度只在 0.006~0.018 这条极窄的能量带里变化，而写字的能量项要到
        #     0.05 才满分。也就是"手在写"的这一整段（0.018~0.05），静止度恒为 0、
        #     反而失去了区分力，只剩时长项在两边同时给分。
        #
        # 所以修法不是调权重，而是**换角色**：运动必须先"足够静止"，
        # 阅读才谈得上时长 —— 静止从"加分项"变成"必要条件"（具备否决权）。
        # 这样"在写"时阅读分会真的被压下去，"在看书"时又不受影响。
        read_score = 0.0
        read_ev = []
        book_present = presence.is_present("book", min_duration=0.5)
        paper_ctx = interactions.paper_contact or book_present

        if paper_ctx:
            dur = interactions.paper_duration
            duration_term = _ramp(dur, 0.8, 2.5)
            # 静止度：只看"手有多静"这一件事（用于给阅读分定档）。
            static_term = _fall(max_energy, self.read_energy_static, self.read_energy_max)
            # 静止**门限**：手明显在动就直接否决阅读。它只做"够不够静"的判决，
            # 不参与加权 —— 否则会和上面的 static_term 叠加（同一件事扣两次），
            # 在过渡带里留下"阅读分剩一点、写字分还没起来"的窄缝，反而制造抖动。
            static_ok = max_energy <= self.read_static_grace

            if has_hands:
                if static_ok:
                    read_score = 0.45 * duration_term + 0.55 * static_term
                    read_ev.append(
                        f"书写区停留 {dur:.1f}s / 静止度 {static_term:.2f}"
                    )
                else:
                    # 手在动 → 不可能是在看书。阅读分归零，把位置让给写字。
                    read_score = 0.0
                    read_ev.append(
                        f"手在动（能量 {max_energy:.3f} > "
                        f"{self.read_static_grace:.3f}）→ 不像在看书"
                    )
            else:
                # 手离开但书还在（低头看书/翻页间隙）。
                # 这是真正需要"翻页宽限"的场景 —— 手不在，谈不上能量门限，
                # 只能靠"书还在 + 刚才在读"来维持，所以给一个较低的常数底分。
                read_score = 0.45 * duration_term + 0.25
                read_ev.append("书在画面内但手已离开")

            if book_present:
                read_ev.append("检测到 book")
        scores[Behavior.READING] = read_score
        evidence[Behavior.READING] = read_ev

        # ---------- 4.5) 打字动作签名 ----------
        # 「手在键盘区」只是位置、不是证据。真正的打字必须过四个量的
        # 几何平均（方向反转 / 脉冲波动 / 幅度收敛 / 有实质动作）——
        # 公式与理由见 `typing_signature()`（唯一定义处，标定工具共用）。
        _signature = self.typing_signature(motions)
        typing_terms = _signature["terms"]
        typing_score = _signature["score"]
        typing_evident = typing_score >= self.comp_typing_min

        # ---------- 5) 电脑学习 ----------
        # 判据链：**检出键盘/笔记本物体 → 手与它发生交互 → 正在做打字动作**。
        # 位置（keyboard ROI）只做召回兜底，而且兜底也必须过打字签名 ——
        # 所以「位置」永远无法单独判定电脑学习。
        comp_score = 0.0
        comp_ev = []
        laptop_present = presence.is_present("laptop", min_duration=0.5)
        keyboard_present = presence.is_present("keyboard", min_duration=0.5)

        obj_ok = bool(
            interactions.keyboard_object_contact or interactions.laptop_contact
        )
        pos_ok = bool(interactions.keyboard_roi_contact)
        contact_ok = obj_ok or (pos_ok and not self.comp_require_object)

        # 打字证据不足时把分数按比例压向 0（而不是硬门限，避免抖动）
        typing_gate = _ramp(
            typing_score, self.comp_typing_min * 0.6, self.comp_typing_min
        )

        if has_hands and contact_ok:
            if obj_ok:
                dur = max(interactions.keyboard_object_duration, 0.0)
            else:
                dur = max(interactions.keyboard_duration, 0.0)

            duration_term = _ramp(dur, 0.5, 2.0)
            comp_score = typing_gate * (0.55 * duration_term + 0.45 * typing_score)

            if obj_ok:
                comp_ev.append(f"键盘物体接触 {dur:.1f}s")
                if interactions.laptop_contact:
                    comp_ev.append("手与 laptop 交互")
                if keyboard_present:
                    comp_ev.append("检测到 keyboard")
                if laptop_present:
                    comp_ev.append("检测到 laptop")
            else:
                # 只有位置命中、一个键盘物体都没检出 —— 证据更弱，折价。
                # 这一条的存在就是为了让「物体检测」真的能改变输出。
                comp_score *= self.comp_position_only_penalty
                comp_ev.append(f"仅位置命中（未检出键盘物体）{dur:.1f}s，已折价")

            comp_ev.append(
                "打字签名 "
                + " ".join(f"{k} {v:.2f}" for k, v in typing_terms.items())
                + f" → {typing_score:.2f}"
            )
            if not typing_evident:
                comp_ev.append(
                    f"打字证据不足（{typing_score:.2f} < {self.comp_typing_min:.2f}）"
                    "，不能判为电脑学习"
                )
        elif has_hands and typing_evident:
            # 手在，动作本身**很像打字**（签名已达标），但位置与物体两条通路
            # 都不成立：手既不在键盘框里，也没检出键盘/笔记本物体。
            # 这正是用户最常问的一句 ——「我明明在敲键盘，为什么检测不出来」。
            # 不解释的话，面板对电脑学习一个字的证据都没有，看起来就是"没装"。
            # 只补证据、不动分数：此时确实无法确认位置，如实说明即可。
            comp_ev.append(
                f"打字动作成立（{typing_score:.2f}），但手不在键盘区、"
                "也未检出键盘/笔记本物体 → 位置无法确认，暂不判电脑学习"
                "（把键盘框对准键盘 / 让键盘进入画面可解决）"
            )
        scores[Behavior.COMPUTER_STUDY] = comp_score
        evidence[Behavior.COMPUTER_STUDY] = comp_ev

        # ---------- 6) 小动作 ----------
        # 这是唯一一个「有正向证据反而应该让位」的行为：键击本身就是高频
        # 方向反转 + 大速度波动，光看幅度/方向，打字和小动作长得一模一样。
        # 所以打字签名成立、且手确实在键盘/笔记本上时，小动作让位。
        # 反过来：**手在键盘区大幅乱挥**过不了打字签名，小动作照常触发 ——
        # 不再出现「电脑学习 + 小动作」同时点亮。
        fid_score = 0.0
        fid_ev = []
        if has_hands and not interactions.paper_contact and not interactions.phone_contact:
            if typing_evident and self.comp_suppress_fidget and contact_ok:
                fid_ev.append(
                    f"打字动作成立（{typing_score:.2f}），有目的的动作，不判小动作"
                )
            else:
                range_term = _ramp(max_range_3s, self.fid_range_lo, self.fid_range_hi)
                dir_term = _ramp(max_dir, self.fid_dir_min, self.fid_dir_min * 2.5)

                # 手明显跑出桌面也算（撑着下巴、转笔、摸头发）
                off_desk_term = 0.35 if interactions.hands_off_desk else 0.0

                fid_score = min(
                    1.0,
                    0.55 * range_term + 0.35 * dir_term + off_desk_term,
                )
                if fid_score > 0.05:
                    fid_ev.append(
                        f"3s 幅度 {max_range_3s:.3f} / 方向变化 {max_dir:.2f}/s"
                    )
                    if interactions.hands_off_desk:
                        fid_ev.append("手离开桌面区域")

                # 能量下限：手基本没动就不是"小动作"。缺了这条，"手压着
                # 键盘发呆"会因为关键点抖动拿到 0.35 的分，被判成分心。
                if max_energy < self.fid_energy_min:
                    fid_score = 0.0
                    if fid_ev:
                        fid_ev.append(
                            f"但运动能量 {max_energy:.4f} < {self.fid_energy_min}"
                            "（几乎没动），不算小动作"
                        )
        scores[Behavior.FIDGETING] = fid_score
        evidence[Behavior.FIDGETING] = fid_ev

        # ---------- 7) 静止（手在，但什么也不做）----------
        idle_score = 0.0
        idle_ev = []
        if has_hands:
            quiet = _fall(max_energy, self.read_energy_static, self.read_energy_max * 1.5)
            on_desk = any(interactions.hand_on_desk.values())

            # 手已经落在某个有意义的区域（书写区 / 键盘 / 手机）里时，
            # 「静止」不应该抢戏 —— 手压着书不动是**阅读**，不是发呆。
            # 没有这一条的话，静止的手会先被判成 IDLE，阅读要等两秒
            # 才能翻盘，置信度也会被稀释。
            contextual = (
                interactions.paper_contact
                or interactions.keyboard_contact
                or interactions.phone_contact
            )

            idle_score = (
                0.65 * quiet
                * (0.6 if not on_desk else 1.0)
                * (0.35 if contextual else 1.0)
            )

            if quiet > 0.5:
                idle_ev.append(f"手部静止 (能量 {max_energy:.3f})")
                if contextual:
                    idle_ev.append("但手在有效区域内，不判发呆")

            # 手在键盘/笔记本上、但打字签名不成立 —— 这是最容易被误解的
            # 一种情况（"我明明在键盘上，为什么不算电脑学习"）。把原因写进
            # 证据里，否则界面只显示"静止"，用户看不出是动作证据不足。
            if contact_ok and not typing_evident:
                idle_ev.append(
                    f"手在键盘/笔记本上，但打字签名 {typing_score:.2f} 不足"
                    f"（需 ≥{self.comp_typing_min:.2f}）→ 不判电脑学习"
                )
        scores[Behavior.IDLE] = idle_score
        evidence[Behavior.IDLE] = idle_ev

        # ---------- 8) 归一化 & 决出 top ----------
        # 相机没帧时不对外输出任何"分心"结论
        if not camera_ok:
            scores = {b: 0.0 for b in self._score_keys}
            evidence[Behavior.UNKNOWN] = ["摄像头无画面"]

        total = sum(scores.values())
        if total > 1e-6:
            normalized = {b: s / total for b, s in scores.items()}
        else:
            normalized = {b: 0.0 for b in scores}

        verdict.scores = normalized
        verdict.evidence = evidence

        # 原始分（未归一化）用于阈值判断，归一化分用于展示。
        # 平分时按 `_TIE_BREAK_ORDER` 显式裁决，不依赖 dict 的插入顺序。
        # 注意用 `(分数, -优先级)` 配 `max`：`max` 取**最大**的键，所以分数要
        # 正向写、优先级要取负 —— 写成 `(-分数, 优先级)` 会变成"选最小分"。
        raw_top = (
            max(scores.items(), key=lambda kv: (kv[1], -_tie_rank(kv[0])))
            if scores else (Behavior.UNKNOWN, 0.0)
        )

        if raw_top[1] < 0.25:
            verdict.top = Behavior.UNKNOWN
            verdict.confidence = raw_top[1]
        else:
            verdict.top = raw_top[0]
            verdict.confidence = min(1.0, raw_top[1] / max(total, 1e-6) * 1.6) if total else 0.0

        verdict.raw_scores = scores
        return verdict
