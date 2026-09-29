"""Hand-Object Interaction Engine。

把"手在哪"和"物体在哪"合成一句话：**手正在操作什么**。

输入：一只手 + 一组物体 + 桌面标定（ROI）
输出：具名交互（拿着手机 / 手在书写区 / 手在键盘上 / 手在桌外 ……）

设计要点：
  * **检测为主、位置为先验**。ROI 只知道"这个位置通常是 X"，不知道那里到底
    有没有 X。所以每一路都**分开报**（`*_object_*` / `*_roi_*`），由行为层
    决定谁有定罪权 —— 用无条件的 `or` + `max()` 把两路合并，会让「检测」这条
    分支永远无法单独改变结果（键盘与手机都栽在这上面过，见 S6）；
  * **书本**经常检不出来，但纸笔一定会落在 `paper` ROI 里 —— 纸这一路仍以
    ROI 为主：写字/阅读是低风险判定，漏判的代价比误判更大；
  * **手机相反**。它原本被当作"见到就基本能定罪"的证据，但那是说**检出手机
    物体**；而"手落在 phone 矩形里"根本不是证据。冒认手机的代价极高（凭空
    记一笔分心、触发提醒、拉低专注分），所以手机默认**必须检出手机物体**
    才定罪，位置单独不作数；
  * 每一路时长都会在窄区域内抖动时被 `InteractionTracker` 的 break_grace
    兜住（丢一两帧不清零）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..features.interaction_features import InteractionTracker, measure

# 需要按 ROI 计时的区域
_ROI_DWELL_KEYS = ("paper", "phone", "keyboard")


@dataclass
class InteractionReport:
    """一帧的交互结论。"""

    timestamp: float = 0.0

    # 手 → ROI
    hand_in_study: dict = field(default_factory=dict)        # {hand_label: bool}
    hand_in_paper_roi: dict = field(default_factory=dict)
    hand_in_keyboard_roi: dict = field(default_factory=dict)
    hand_in_phone_roi: dict = field(default_factory=dict)
    hand_on_desk: dict = field(default_factory=dict)

    # 手 → 物体（检测驱动）
    contact_objects: dict = field(default_factory=dict)      # {hand_label: set}

    # ---- 汇总结论 ----
    phone_contact: bool = False
    phone_duration: float = 0.0
    phone_hands: list = field(default_factory=list)

    # 手机的两条通路同样**必须分开报**（理由同键盘）：位置单独不足以定罪。
    phone_object_contact: bool = False   # 检出 phone 物体且手与它几何接触
    phone_roi_contact: bool = False      # 仅仅是掌心落在 phone 矩形内
    phone_object_duration: float = 0.0
    phone_roi_duration: float = 0.0

    paper_contact: bool = False
    paper_duration: float = 0.0
    paper_hands: list = field(default_factory=list)

    keyboard_contact: bool = False
    keyboard_duration: float = 0.0

    # 键盘的两条通路**必须分开报**，否则「位置」会把「检测」吞掉：
    # 只要手落在 keyboard 矩形里 keyboard_contact 就成立，物体检测这条
    # 分支永远无法单独改变结果（`or` + `max()` 的组合会把它的时长也追平）。
    # 现在由行为层按「检测为主、位置为备」重新组合。
    keyboard_object_contact: bool = False    # 检出 keyboard 物体且手与它几何接触
    keyboard_roi_contact: bool = False       # 仅仅是掌心落在 keyboard 矩形内
    keyboard_object_duration: float = 0.0
    keyboard_roi_duration: float = 0.0

    laptop_contact: bool = False
    book_contact: bool = False

    # 手是否在"桌面"之外（手举起来 / 放在腿上 / 撑着下巴）
    hands_off_desk: list = field(default_factory=list)

    # 原始测量，供 Debug 视图画线
    measures: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phone_contact": self.phone_contact,
            "phone_duration": round(self.phone_duration, 2),
            "phone_object_contact": self.phone_object_contact,
            "phone_object_duration": round(self.phone_object_duration, 2),
            "phone_roi_contact": self.phone_roi_contact,
            "paper_contact": self.paper_contact,
            "paper_duration": round(self.paper_duration, 2),
            "keyboard_contact": self.keyboard_contact,
            "keyboard_duration": round(self.keyboard_duration, 2),
            "keyboard_object_contact": self.keyboard_object_contact,
            "keyboard_object_duration": round(self.keyboard_object_duration, 2),
            "keyboard_roi_contact": self.keyboard_roi_contact,
            "laptop_contact": self.laptop_contact,
            "book_contact": self.book_contact,
        }


class HandObjectEngine:
    """手-物交互引擎。"""

    def __init__(self, config, calibration):
        behavior = config.behavior

        self.distance_px = float(behavior.get("interaction_distance_px", 90))
        self.overlap_min = float(behavior.get("interaction_overlap_min", 0.05))

        # 位置先验总开关。False（默认）时所有 ROI 判定被中性化：
        #   * paper/keyboard/phone 的 ROI 命中一律 False —— 写字/阅读的
        #     上下文只剩「检出 book 物体」，键盘/手机只剩物体检测通路；
        #   * desk 判定跳过：hand_on_desk 恒 True（IDLE 不因"不在桌面"
        #     折价）、hands_off_desk 恒空（小动作不因位置加成）。
        # 行为融合层不需要任何改动 —— 它读到的"位置信号"在这台机器上
        # 永远是中性的，结论完全由物体检测与手部动作特征驱动。
        # 运行时可在菜单「标定 → 启用位置先验」切换本属性。
        self.use_position_prior = bool(behavior.get("use_position_prior", False))

        self.calibration = calibration
        self.contacts = InteractionTracker(break_grace=0.6)

        # ROI 停留计时（每个区域的起始时刻）
        self._roi_start: dict[str, float] = {}
        self._roi_duration: dict[str, float] = {k: 0.0 for k in _ROI_DWELL_KEYS}

        self._last_report = InteractionReport()

    # ------------------------------------------------------------------

    def update(
        self,
        hands: list,
        tracked_objects: list,
        now: float,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> InteractionReport:
        report = InteractionReport(timestamp=now)
        calib = self.calibration

        # ---------- 1) 手 → ROI ----------
        use_pos = self.use_position_prior
        for hand in hands:
            label = hand.label
            cx, cy = hand.palm_center

            if use_pos:
                report.hand_in_study[label] = calib.contains("study", cx, cy)
                report.hand_in_paper_roi[label] = calib.contains("paper", cx, cy)
                report.hand_in_keyboard_roi[label] = calib.contains("keyboard", cx, cy)
                report.hand_in_phone_roi[label] = calib.contains("phone", cx, cy)

                on_desk = calib.contains("desk", cx, cy)
                report.hand_on_desk[label] = on_desk
                if not on_desk:
                    report.hands_off_desk.append(label)
            else:
                # 位置先验关闭：ROI 信号全部中性化（见 __init__ 注释）。
                report.hand_in_study[label] = False
                report.hand_in_paper_roi[label] = False
                report.hand_in_keyboard_roi[label] = False
                report.hand_in_phone_roi[label] = False
                report.hand_on_desk[label] = True

        # ---------- 2) 手 → 物体（几何测量）----------
        all_measures = []
        contact_map: dict[str, set] = {h.label: set() for h in hands}

        for hand in hands:
            for obj in tracked_objects:
                if obj.label == "person":       # 人对人无意义
                    continue

                m = measure(
                    hand, obj,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    distance_px_threshold=self.distance_px,
                    overlap_min=self.overlap_min,
                )
                all_measures.append(m)

                if m.contact:
                    contact_map.setdefault(hand.label, set()).add(obj.label)

        report.measures = all_measures
        report.contact_objects = contact_map

        # ---------- 3) 更新接触窗口 ----------
        self.contacts.update(all_measures, now)

        # ---------- 4) 逐条汇总 ----------

        # --- 手机：物体检测 / 位置命中，两条通路分开报 ---
        # 检出 phone 物体**并且**手的锚点与它几何接触 —— 这才是「手正在玩
        # 手机」的直接证据，可以定罪。
        phone_obj_hands = [
            h.label for h in hands
            if "phone" in contact_map.get(h.label, set())
        ]
        # 仅仅掌心落在 phone 矩形内。这是**先验**，不是证据：它只知道"这个
        # 位置通常是放手机的地方"，不知道那里到底有没有手机 —— 看书、敲键盘、
        # 托腮都可能把掌心送进这个框里去。行为层默认不让它单独定罪。
        phone_roi_hands = [
            h.label for h in hands
            if report.hand_in_phone_roi.get(h.label, False)
        ]
        report.phone_object_contact = bool(phone_obj_hands)
        report.phone_roi_contact = bool(phone_roi_hands)
        report.phone_hands = phone_obj_hands or phone_roi_hands
        # 兼容字段：任一成立。**不再代表可以据此定罪**。
        report.phone_contact = bool(report.phone_hands)

        # --- 书写区（paper ROI 或 book 接触）---
        report.paper_hands = [
            h.label for h in hands
            if report.hand_in_paper_roi.get(h.label, False)
            or "book" in contact_map.get(h.label, set())
        ]
        report.paper_contact = bool(report.paper_hands)

        # --- 键盘：物体检测 / 位置命中，两条通路分开报 ---
        # 检出 keyboard 物体 **并且** 手的锚点与它几何接触（距离<阈值 或
        # 关键点入框）—— 这才是「手正在操作键盘」的直接证据。
        kb_object_hands = [
            h.label for h in hands
            if "keyboard" in contact_map.get(h.label, set())
        ]
        # 仅仅掌心落在 keyboard 矩形内。这是**先验**，不是证据：它只知道
        # 「这个位置通常是键盘」，不知道那里到底有没有键盘。
        kb_roi_hands = [
            h.label for h in hands
            if report.hand_in_keyboard_roi.get(h.label, False)
        ]
        report.keyboard_object_contact = bool(kb_object_hands)
        report.keyboard_roi_contact = bool(kb_roi_hands)
        # 兼容字段：任一成立。**不再代表可以据此定罪** —— 行为层还必须再
        # 过一道「打字动作签名」，位置本身永远无法单独判定电脑学习。
        report.keyboard_contact = bool(kb_object_hands or kb_roi_hands)

        # --- 电脑 / 书（仅检测驱动）---
        report.laptop_contact = any("laptop" in c for c in contact_map.values())
        report.book_contact = any("book" in c for c in contact_map.values())

        # ---------- 5) ROI 停留时长 ----------
        # 注意 keyboard / phone 这两路喂的都是 **ROI 命中**（不是与检测取或），
        # 否则这个「ROI 停留」其实在计量物体接触，名字就骗人了。
        self._update_roi_dwell(
            {
                "paper": report.paper_contact,
                "phone": report.phone_roi_contact,
                "keyboard": report.keyboard_roi_contact,
            },
            now,
        )

        # ---------- 6) 持续时长：两条通路各自记账 ----------
        report.phone_object_duration = self._max_contact_duration(
            phone_obj_hands, "phone"
        )
        report.phone_roi_duration = self._roi_duration["phone"]
        report.phone_duration = max(
            report.phone_object_duration, report.phone_roi_duration
        )
        report.paper_duration = max(
            self._max_contact_duration(report.paper_hands, "book"),
            self._roi_duration["paper"],
        )
        report.keyboard_object_duration = self._max_contact_duration(
            kb_object_hands, "keyboard"
        )
        report.keyboard_roi_duration = self._roi_duration["keyboard"]
        report.keyboard_duration = max(
            report.keyboard_object_duration, report.keyboard_roi_duration
        )

        self._last_report = report
        return report

    # ------------------------------------------------------------------

    def _max_contact_duration(self, hand_labels: list, object_label: str) -> float:
        """这批手里，接触该物体最久的时长。"""
        if not hand_labels:
            return 0.0
        return max(
            (self.contacts.duration(h, object_label) for h in hand_labels),
            default=0.0,
        )

    def _update_roi_dwell(self, active: dict, now: float) -> None:
        """维护每个区域的连续停留时长。

        只在真正"开始命中"时记录起点，失手瞬间清零 —— 这里**不加**
        break_grace，因为 ROI 是像素级的确定判断，抖动很小。
        """
        for name, is_active in active.items():
            if is_active:
                if name not in self._roi_start:
                    self._roi_start[name] = now
                self._roi_duration[name] = now - self._roi_start[name]
            else:
                self._roi_start.pop(name, None)
                self._roi_duration[name] = 0.0

    # ------------------------------------------------------------------

    def roi_duration(self, name: str) -> float:
        return self._roi_duration.get(name, 0.0)

    @property
    def last_report(self) -> InteractionReport:
        return self._last_report

    def reset(self) -> None:
        self.contacts.clear()
        self._roi_start.clear()
        self._roi_duration = {k: 0.0 for k in _ROI_DWELL_KEYS}
        self._last_report = InteractionReport()
