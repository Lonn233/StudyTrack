"""桌面学习行为检测与专注陪伴助手 —— 主程序。

    python app.py                      # 正常启动
    python app.py --no-debug           # 不打开调试窗口
    python app.py --no-eyes            # 不显示智能体眼睛（主界面右下角）
    python app.py --calibrate          # 启动时先做桌面标定
    python app.py --camera 1           # 指定摄像头序号
    python app.py --headless --seconds 30   # 无界面跑 30 秒（自检 / 日志）

数据流：

    摄像头 → VisionPipeline（分频：手 15Hz / 物体 5Hz + 跨帧跟踪）
           → BehaviorCore（轨迹 → 交互 → 打分 → 时序 → 状态机 → 会话）
           → UI（Dashboard / DebugView / 内嵌 8bit 智能体眼睛）

本文件只负责三件事：**命令行、Qt 装配、把两者接到一起**。所有判断逻辑
都在 `study_assistant/core.py` 里，验收测试直接驱动同一个类。

线程模型：**单线程**。视觉推理 + 逻辑 + UI 都在 Qt 主线程里按 QTimer
驱动。理由：
  * MediaPipe 的 landmarker 不是线程安全的，跨线程调只会更难查；
  * 4 核的树莓派上并行推理总吞吐反而下降（内存带宽是瓶颈）；
  * 单线程让"某一帧为什么这么判"可以完整复现，调试成本低得多。
将来若单帧耗时超过 100ms，把 VisionPipeline 搬到 QThread 即可 ——
接口已经解耦，改动只在这个文件里。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="桌面学习行为检测与专注陪伴助手",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"),
                        help="配置文件路径")
    parser.add_argument("--camera", type=int, default=None, help="摄像头序号")
    parser.add_argument("--backend", default=None,
                        choices=["auto", "opencv", "picamera2"], help="摄像头后端")
    parser.add_argument("--calibrate", action="store_true",
                        help="启动前先做一次桌面标定")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="无交互地写入默认 ROI（无人值守场景）")

    parser.add_argument("--no-debug", action="store_true", help="不打开调试窗口")
    parser.add_argument("--no-eyes", action="store_true", help="不显示智能体眼睛（主界面右下角内嵌）")
    parser.add_argument("--no-db", action="store_true", help="不写数据库")

    parser.add_argument("--headless", action="store_true",
                        help="无图形界面运行（自检 / 服务器）")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="headless 模式下运行多少秒后自动退出（0=一直跑）")

    parser.add_argument("--dump-every", type=float, default=0.0,
                        help="每 N 秒把调试面板写到 out/debug_dump.txt")
    parser.add_argument("--save-frame", default="",
                        help="退出时把带标注的画面存到该路径")
    parser.add_argument("--log-every", type=float, default=5.0,
                        help="headless 模式下每 N 秒打印一行状态")

    parser.add_argument("--screenshot", default="",
                        help="启动 N 秒后把界面截图保存到该路径（用于自动化验收）")
    parser.add_argument("--screenshot-after", type=float, default=8.0,
                        help="等待多少秒再截图")
    parser.add_argument("--exit-after", type=float, default=0.0,
                        help="图形界面运行 N 秒后自动退出（0=一直运行）")

    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# 应用
# ----------------------------------------------------------------------


class StudyAssistantApp:
    """把视觉管线、行为核心、界面装配在一起。"""

    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.calibration = None
        self.pipeline = None
        self.core = None
        self.db = None
        self.notifications = None

        self.dashboard = None
        self.debug_view = None
        self.eyes = None
        self.tray = None

        self.debug_options = dict(config.ui.get("debug", {}) or {})

        for key, default in (
            ("show_roi", True),
            ("show_hands", True),
            ("show_objects", True),
            ("show_interaction_lines", True),
            ("show_trajectory", True),
        ):
            self.debug_options.setdefault(key, default)

        self.running = False
        self.start_t = 0.0

        self.missed_frames = 0
        self.last_ok_frame_t = 0.0
        self.camera_ok = True

        self.board_info = ""
        self.last_vision_frame = None
        self.last_events: list = []

        self._last_dump_t = 0.0
        self._last_render_t = 0.0
        self._render_interval = 1.0 / 15.0     # 界面刷新上限 15Hz

        self.frame_width = int(config.camera.get("width", 640))
        self.frame_height = int(config.camera.get("height", 480))

        self._msg_box = None

    # ------------------------------------------------------------------
    # 便捷访问
    # ------------------------------------------------------------------

    @property
    def last_snapshot(self):
        return self.core.last.snapshot if self.core else None

    @property
    def last_estimate(self):
        return self.core.last.estimate if self.core else None

    @property
    def last_interactions(self):
        return self.core.last.interactions if self.core else None

    @property
    def hand_tracker(self):
        return self.core.hand_tracker if self.core else None

    @property
    def session(self):
        return self.core.session if self.core else None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def build(self) -> None:
        from study_assistant.camera.camera_backend import create_camera
        from study_assistant.core import BehaviorCore
        from study_assistant.notifications.notification_manager import NotificationManager
        from study_assistant.storage.database import Database
        from study_assistant.ui.calibration_view import load_calibration, run_calibration
        from study_assistant.vision.vision_pipeline import VisionPipeline

        # --- 摄像头 ---
        camera_cfg = dict(self.config.data.get("camera", {}))

        if self.args.camera is not None:
            camera_cfg["index"] = self.args.camera
        if self.args.backend is not None:
            camera_cfg["backend"] = self.args.backend

        camera = create_camera({"camera": camera_cfg})

        # --- 标定 ---
        if self.args.auto_calibrate:
            from study_assistant.ui.calibration_view import auto_calibration

            self.calibration = auto_calibration(self.config, camera)
        elif self.args.calibrate:
            self.calibration = run_calibration(self.config, camera)

            if self.calibration is None:
                self.calibration = load_calibration()
        else:
            self.calibration = load_calibration()

        if not self.calibration.completed:
            print("[提示] 尚未做桌面标定，行为判定会保守一些。")
            print("       运行 `python app.py --calibrate` 可以补做。")

        # --- 视觉 ---
        self.pipeline = VisionPipeline(self.config, camera=camera, root=ROOT)

        for warning in self.pipeline.warnings:
            print(f"[警告] {warning}")

        from study_assistant.vision.board_profile import describe

        self.board_info = describe(self.pipeline.board)

        # --- 存储 ---
        if not self.args.no_db and self.config.storage.get("enabled", True):
            db_path = self.config.user_data_dir / self.config.storage.get(
                "database", "study_assistant.sqlite3"
            )
            self.db = Database(db_path)
            print(f"[存储] 数据库：{db_path}")

        # --- 通知 ---
        self.notifications = NotificationManager(
            self.config,
            on_notify=self._on_notify,
            on_mood=self._set_mood,
        )

        # --- 行为核心 ---
        self.core = BehaviorCore(
            self.config,
            self.calibration,
            database=self.db,
            notifications=self.notifications,
        )
        self.core.hand_detector_available = self.pipeline.hand_detector is not None

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.pipeline.start()

        self.running = True
        self.start_t = time.perf_counter()
        self.last_ok_frame_t = self.start_t

        events = self.core.start(self.start_t)
        self.last_events = events
        self._push_events_to_ui(events)

        print(f"[启动] {self.board_info}")

    def tick(self) -> None:
        """跑一帧。"""
        now_perf = time.perf_counter()
        vision_frame = self.pipeline.step()

        # --- 摄像头无帧 ---
        if vision_frame is None:
            self.missed_frames += 1

            if (now_perf - self.last_ok_frame_t) > 2.0:
                self.camera_ok = False

            # 让逻辑继续跑（会平滑进入 UNCERTAIN），不要卡死界面
            self.core.update(
                hands=[], tracked_objects=[], now=now_perf,
                frame_width=self.frame_width, frame_height=self.frame_height,
                camera_ok=self.camera_ok, hands_fresh=False, objects_fresh=False,
            )
            self._after_core_update()
            return

        self.missed_frames = 0
        self.last_ok_frame_t = now_perf
        self.camera_ok = True

        self.frame_height, self.frame_width = vision_frame.frame.shape[:2]
        self.last_vision_frame = vision_frame

        self.core.update(
            hands=vision_frame.hands.hands,
            tracked_objects=vision_frame.tracked.objects,
            now=vision_frame.now,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            camera_ok=True,
            hands_fresh=vision_frame.hands_fresh,
            objects_fresh=vision_frame.objects_fresh,
        )
        self._after_core_update()

        # --- 渲染（限频 15Hz）---
        if (now_perf - self._last_render_t) >= self._render_interval:
            self._last_render_t = now_perf
            self._render(vision_frame)

        # --- 定时 dump ---
        if self.args.dump_every > 0 and (now_perf - self._last_dump_t) >= self.args.dump_every:
            self._last_dump_t = now_perf
            self._dump_debug()

    def _after_core_update(self) -> None:
        """核心更新之后的公共动作：表情、事件分发。"""
        update = self.core.last

        self._sync_eyes(update.snapshot)

        if update.events:
            self.last_events = update.events
            self._push_events_to_ui(update.events)

            for event in update.events:
                if event.kind == "away_start":
                    self._set_mood("AWAY")
                elif event.kind == "away_end":
                    self._set_mood("CURIOUS")
                elif event.kind == "session_end":
                    self.notifications.notify(
                        event.message, level=2, title="本次学习", kind="session_end"
                    )

    # ------------------------------------------------------------------

    def _push_events_to_ui(self, events) -> None:
        if self.dashboard is not None and events:
            self.dashboard.update_view(events=events, session=self.session)

    def _render(self, vision_frame) -> None:
        from study_assistant.ui.overlay import render_debug_frame

        trajectories = self.core.hand_tracker.trajectories(
            window=3.0, now=vision_frame.now
        )

        annotated = render_debug_frame(
            vision_frame,
            calibration=self.calibration,
            interactions=self.core.last.interactions,
            trajectories=trajectories,
            options=self.debug_options,
        )

        if self.dashboard is not None:
            self.dashboard.update_view(
                annotated_frame=annotated,
                state_snapshot=self.core.last.snapshot,
                estimate=self.core.last.estimate,
                stats=self.core.snapshot_stats(vision_frame.now),
                session=self.core.session,
                vision_frame=vision_frame,
            )

        if self.debug_view is not None and self.debug_view.isVisible():
            self.debug_view.update_debug(
                vision_frame,
                calibration=self.calibration,
                interactions=self.core.last.interactions,
                trajectories=trajectories,
                estimate=self.core.last.estimate,
                snapshot=self.core.last.snapshot,
                session=self.core.session,
                hands_tracker=self.core.hand_tracker,
                board_info=self.board_info,
            )

    # ------------------------------------------------------------------

    def _dump_debug(self) -> None:
        """把管线内部状态写到 out/（headless 也能用，便于远程验收）。"""
        from study_assistant.ui.debug_view import build_feature_text, build_perf_text

        out_dir = ROOT / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        vf = self.last_vision_frame
        sections = [f"# Debug dump @ {time.strftime('%Y-%m-%d %H:%M:%S')}"]

        if vf is not None:
            sections.append("\n## 性能\n")
            sections.append(build_perf_text(vf, self.board_info))

            sections.append("\n\n## 特征\n")
            sections.append(
                build_feature_text(
                    vf,
                    self.core.last.interactions,
                    self.core.last.estimate,
                    self.core.last.snapshot,
                    self.core.session,
                    self.core.hand_tracker,
                )
            )
        else:
            sections.append("\n（还没有可用帧）")

        snapshot = self.core.last.snapshot

        if snapshot is not None:
            sections.append("\n\n## 快照\n")
            sections.append(f"状态: {snapshot.state.value}")
            sections.append(f"行为: {snapshot.behavior.value}")
            sections.append(f"原因: {snapshot.reason}")
            sections.append(f"离席信号: 手 {self.core.last.hands_absent_for:.1f}s "
                            f"/ 人 {self.core.last.person_absent_for:.1f}s "
                            f"/ 人在场={self.core.last.person_present}")

        (out_dir / "debug_dump.txt").write_text("\n".join(sections), encoding="utf-8")

        (out_dir / "state_snapshot.txt").write_text(
            "\n".join(
                [
                    f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"统计: {self.core.snapshot_stats().as_dict()}",
                ]
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------

    def _set_mood(self, mood: str) -> None:
        if self.eyes is not None:
            self.eyes.set_mood(mood)

    def _sync_eyes(self, snapshot) -> None:
        """由状态 + 行为决定悬浮窗表情。"""
        if self.eyes is None or snapshot is None:
            return

        from study_assistant.behavior.states import (
            BEHAVIOR_TO_MOOD,
            STATE_TO_MOOD,
            Behavior,
            FocusState,
        )

        behavior = snapshot.behavior

        mood = (
            BEHAVIOR_TO_MOOD.get(behavior)
            if behavior is not Behavior.UNKNOWN
            else STATE_TO_MOOD.get(snapshot.state)
        )

        # 分心持续时间越长，表情越不耐烦（升级式表现）
        if snapshot.state is FocusState.DISTRACTED:
            distraction = self.core.current_distraction

            if distraction >= self.core.session.alert_seconds:
                mood = "ANNOYED"
            elif distraction >= self.core.session.soft_seconds:
                mood = "SUSPICIOUS"
            else:
                mood = "WORRIED"

        if mood:
            self.eyes.set_mood(mood)

    def _on_notify(self, title: str, message: str, level: int) -> None:
        """Qt 托盘气泡（未接 Qt 时退化为打印）。"""
        if self.tray is not None:
            try:
                self.tray.showMessage(title, message)
                return
            except Exception:
                pass

        print(f"[提醒 L{level}] {title} — {message}")

    # ------------------------------------------------------------------

    def shutdown(self, end_session: bool = True) -> None:
        self.running = False

        if end_session and self.core is not None and self.core.session.active:
            stats, _events = self.core.end()
            print(
                f"[结束] 有效专注 {stats.focused_seconds:.0f}s / "
                f"专注分 {stats.focus_score:.0f} / 切换 {stats.transitions} 次"
            )

        if self.pipeline is not None:
            self.pipeline.close()

        if self.db is not None:
            self.db.close()


# ----------------------------------------------------------------------
# Qt 装配
# ----------------------------------------------------------------------


def run_gui(app: StudyAssistantApp, args) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

    from study_assistant.behavior.focus_score import format_duration, grade
    from study_assistant.ui.dashboard import Dashboard
    from study_assistant.ui.debug_view import DebugView

    qt_app = QApplication.instance() or QApplication(sys.argv)

    from study_assistant.ui.theme import apply as apply_theme

    apply_theme(qt_app)

    dashboard = Dashboard(
        app.config,
        calibration=app.calibration,
        options=app.debug_options,
    )
    dashboard.show()
    app.dashboard = dashboard

    # --- 智能体眼睛：内嵌在右侧控制板底部 ---
    # 旧版是独立悬浮窗；新版眼睛住在主界面里，`app.eyes` 指向内嵌实例，
    # `_set_mood / _sync_eyes` 等既有逻辑无需改动。
    if not args.no_eyes:
        app.eyes = dashboard.eyes_widget
    else:
        dashboard.eyes_widget.setVisible(False)

    # --- 调试窗口 ---
    if not args.no_debug and app.config.ui.get("debug", {}).get("enabled", True):
        debug_view = DebugView(app.debug_options)
        debug_view.options_changed.connect(lambda opts: app.debug_options.update(opts))
        debug_view.show()
        app.debug_view = debug_view

    # --- 托盘 ---
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(QIcon(), qt_app)
        tray.setToolTip("Study Assistant")
        tray.show()
        app.tray = tray

    # --- 交互回调 ---
    def apply_pause(paused: bool) -> None:
        app.core.set_paused(paused)
        dashboard.set_paused_state(paused)

    def on_pause():
        apply_pause(True)

    def on_start():
        # 暂停中 → 继续；否则（会话已结束/从未开始）开一段新的
        if dashboard.paused:
            apply_pause(False)
        else:
            new_events = app.core.reset()
            app._push_events_to_ui(new_events)

    def on_restart():
        # 先收尾当前会话（落库），再开一段新的
        app.core.end()
        new_events = app.core.reset()
        app._push_events_to_ui(new_events)

    def end_session():
        stats, events = app.core.end()
        letter, comment = grade(stats.focus_score)

        QMessageBox.information(
            dashboard,
            "本次学习总结",
            f"有效专注：{format_duration(stats.focused_seconds)}\n"
            f"分心：{format_duration(stats.distracted_seconds)}\n"
            f"离席：{format_duration(stats.away_seconds)}\n"
            f"专注分：{stats.focus_score:.0f}（{letter} · {comment}）\n"
            f"状态切换：{stats.transitions} 次",
        )
        app._push_events_to_ui(events)

    def calibrate():
        from study_assistant.ui.calibration_view import run_calibration

        app.pipeline.camera.release()
        result = run_calibration(app.config, app.pipeline.camera)
        app.pipeline.camera.open()

        if result is not None:
            app.calibration = result
            app.core.engine.calibration = result
            dashboard.set_calibration(result)

    def toggle_debug():
        if app.debug_view is not None:
            app.debug_view.setVisible(not app.debug_view.isVisible())

    def on_options(opts: dict) -> None:
        # 叠加图层菜单（ROI/骨架/物体框/连线/轨迹）→ 渲染选项
        app.debug_options.update(opts)

    def take_break():
        app.core.session.timer.take_break_now()
        app.notifications.suggest_break("休息一下吧，回来继续")

    def on_quit_menu():
        qt_app.quit()

    # --- 摄像头切换 ---
    def refresh_camera_menu() -> None:
        from study_assistant.camera.camera_enum import enumerate_cameras

        # skip_index：当前正被自己使用的索引不做探测（探测它=自己占着自己）
        app.camera_list = enumerate_cameras(skip_index=app.current_camera_index)
        dashboard.refresh_camera_menu(app.camera_list, app.current_camera_index)

    def on_camera_selected(index: int) -> None:
        """菜单选择摄像头 → 热切换；失败时旧摄像头保持不动并写事件流。"""
        if app.pipeline is None or index == app.current_camera_index:
            return
        try:
            app.pipeline.switch_camera(index)
        except Exception as exc:
            name = next(
                (c.name for c in app.camera_list if c.index == index),
                f"索引 {index}",
            )
            dashboard.post_system_event(
                f"切换到「{name}」失败：{exc}", level=2
            )
            print(f"[摄像头] 切换到 {name}(索引 {index}) 失败：{exc}")
            return
        app.current_camera_index = index
        name = next(
            (c.name for c in app.camera_list if c.index == index), f"索引 {index}"
        )
        dashboard.post_system_event(f"已切换摄像头：{name}")
        print(f"[摄像头] 已切换到 {name}")
        refresh_camera_menu()

    def on_camera_rescan() -> None:
        refresh_camera_menu()
        dashboard.post_system_event("已重新扫描摄像头")
        print("[摄像头] 已重新扫描")

    dashboard.signal_pause.connect(on_pause)
    dashboard.signal_start.connect(on_start)
    dashboard.signal_reset.connect(on_restart)
    dashboard.signal_end_session.connect(end_session)
    dashboard.signal_calibrate.connect(calibrate)
    dashboard.signal_toggle_debug.connect(toggle_debug)
    dashboard.signal_options_changed.connect(on_options)
    dashboard.signal_break.connect(take_break)
    dashboard.signal_quit.connect(on_quit_menu)
    dashboard.signal_camera_selected.connect(on_camera_selected)
    dashboard.signal_camera_rescan.connect(on_camera_rescan)

    # --- 位置先验开关（判定是否使用标定框）---
    def on_position_prior(on: bool) -> None:
        app.core.engine.use_position_prior = bool(on)
        state = "启用" if on else "关闭"
        dashboard.post_system_event(
            f"已{state}位置先验：{'标定框重新参与判定' if on else '判定只看物体检测与手部动作'}"
        )
        print(f"[判定] 位置先验已{state}")

    dashboard.signal_position_prior_changed.connect(on_position_prior)

    # --- 摄像头菜单初值 ---
    app.current_camera_index = getattr(getattr(app.pipeline, "camera", None), "index", 0)
    app.camera_list = []
    refresh_camera_menu()

    # --- 主循环 ---
    target_fps = app.config.data.get("app", {}).get("target_fps", 30)
    interval_ms = max(10, int(1000 / max(1, target_fps)))

    timer = QTimer()
    timer.setInterval(interval_ms)

    def on_tick():
        try:
            app.tick()
        except Exception as exc:
            import traceback

            traceback.print_exc()

            try:
                app.notifications.notify(f"内部错误：{exc}", level=2, kind="error")
            except Exception:
                pass

    timer.timeout.connect(on_tick)

    app.start()
    timer.start()

    def on_quit():
        timer.stop()
        app.shutdown()

    qt_app.aboutToQuit.connect(on_quit)

    # --- 自动截图 / 自动退出（无人值守验收用）---
    if args.screenshot:
        def take_shot():
            _save_screenshot(app, args.screenshot)

        QTimer.singleShot(int(max(0.5, args.screenshot_after) * 1000), take_shot)

    if args.exit_after > 0:
        def auto_exit():
            write_snapshot(app)
            qt_app.quit()

        QTimer.singleShot(int(max(0.5, args.exit_after) * 1000), auto_exit)

    return qt_app.exec()


def write_snapshot(app: StudyAssistantApp) -> None:
    """把运行状态写到 out/ui_run.txt（自动化判读用）。"""
    out_dir = ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = app.last_snapshot
    stats = app.core.snapshot_stats() if app.core else None
    vision = app.last_vision_frame

    lines = [
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"状态: {snapshot.state.value if snapshot else 'N/A'}",
        f"行为: {snapshot.behavior.value if snapshot else 'N/A'}",
        f"帧序号: {vision.index if vision else 'N/A'}",
        f"帧率: {vision.fps:.1f}" if vision else "帧率: N/A",
        f"轨迹数: {len(vision.tracked.objects) if vision else 0}",
        f"统计: {stats.as_dict() if stats else 'N/A'}",
    ]

    (out_dir / "ui_run.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_screenshot(app: StudyAssistantApp, path: str) -> None:
    """把主界面与悬浮窗各自截图存盘。

    用 `QWidget.grab()` 而不是 Win32 `PrintWindow`：不受窗口遮挡影响，
    也不需要 P/Invoke（本机安全策略会拦内联 C#）。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    saved = []

    if app.dashboard is not None:
        if app.dashboard.grab().save(str(target)):
            saved.append(str(target))

    if app.eyes is not None and app.eyes.parent() is None:
        # 只有独立悬浮窗才单独截图；内嵌在主界面里时主界面截图已包含
        eyes_path = target.with_name(target.stem + "_eyes" + target.suffix)
        if app.eyes.grab().save(str(eyes_path)):
            saved.append(str(eyes_path))

    if app.debug_view is not None and app.debug_view.isVisible():
        dbg_path = target.with_name(target.stem + "_debug" + target.suffix)
        if app.debug_view.grab().save(str(dbg_path)):
            saved.append(str(dbg_path))

    print(f"[截图] 已保存 {len(saved)} 张：{saved}")


def run_headless(app: StudyAssistantApp, args) -> int:
    """无界面运行（自检 / 服务器 / CI）。"""
    app.start()

    deadline = (time.perf_counter() + args.seconds) if args.seconds > 0 else None
    last_log = time.perf_counter()

    try:
        while app.running:
            app.tick()

            now = time.perf_counter()

            if (now - last_log) >= max(0.5, args.log_every):
                last_log = now
                snapshot = app.last_snapshot

                if snapshot is not None:
                    vision = app.last_vision_frame
                    fps = vision.fps if vision else 0.0

                    print(
                        f"[{now - app.start_t:6.1f}s] "
                        f"{snapshot.state.value:<11s} "
                        f"{snapshot.behavior.value:<15s} "
                        f"专注 {app.core.session.focused_seconds:5.0f}s "
                        f"分 {app.core.live_score():5.1f} "
                        f"fps {fps:4.1f}"
                    )

            if deadline is not None and now >= deadline:
                break

            time.sleep(0.004)
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C")
    finally:
        if args.save_frame:
            _save_frame(app, args.save_frame)

        app.shutdown()

    return 0


def _save_frame(app: StudyAssistantApp, path: str) -> None:
    """把当前带标注的画面存盘（用于远程确认摄像头与标注是否正常）。"""
    import cv2

    from study_assistant.ui.overlay import render_debug_frame

    vf = app.last_vision_frame

    if vf is None:
        print("[保存] 没有可用帧")
        return

    annotated = render_debug_frame(
        vf,
        calibration=app.calibration,
        interactions=app.core.last.interactions,
        trajectories=app.core.hand_tracker.trajectories(window=3.0, now=vf.now),
        options=app.debug_options,
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(target), annotated)
    print(f"[保存] {target}  {'OK' if ok else '失败'}")


# ----------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    # 本机 Python 环境的必要开关（见 README「本机环境注意事项」）
    os.environ.setdefault("CODEBUDDY_SAFE_DELETE_ENABLED", "0")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from study_assistant.config import Config

    config = Config.load(args.config)

    app = StudyAssistantApp(config, args)

    try:
        app.build()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\n[致命] 初始化失败：{exc}")
        return 2

    if args.headless:
        return run_headless(app, args)

    return run_gui(app, args)


if __name__ == "__main__":
    sys.exit(main())
