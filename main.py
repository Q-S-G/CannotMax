import csv
import os
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox
import cv2
import keyboard
import numpy as np
import torch
import loadData
import recognize
import math
import pandas as pd
import train
from train import UnitAwareTransformer
from recognize import MONSTER_COUNT, intelligent_workers_debug
from PIL import Image, ImageTk  # 需要安装Pillow库
from sklearn.metrics.pairwise import cosine_similarity


class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Neural Network")
        self.load_history_data()

        self.main_panel = tk.Frame(self.root)
        self.main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 鼠标滚轮滚动历史面板
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        # 运行
        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        # 用户选项
        self.is_invest = tk.BooleanVar(value=False)  # 添加投资状态变量
        self.game_mode = tk.StringVar(value="单人")  # 添加游戏模式变量，默认单人模式
        self.device_serial = tk.StringVar(value=loadData.manual_serial)  # 添加设备序列号变量

        # 数据缓存
        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}
        self.progress_var = tk.StringVar()
        self.main_roi = None

        # 统计
        self.total_fill_count = 0
        self.incorrect_fill_count = 0
        self.start_time = None

        self.load_images()
        self.create_widgets()

        # 历史对局面板
        self.history_visible = False
        self.history_container = tk.Frame(self.root, bd=1, relief="sunken")

        # Canvas & Scrollbars
        self.history_canvas = tk.Canvas(self.history_container, bg="white")
        self.history_vscroll = tk.Scrollbar(
            self.history_container, orient="vertical",
            command=self.history_canvas.yview)
        self.history_hscroll = tk.Scrollbar(
            self.history_container, orient="horizontal",
            command=self.history_canvas.xview)

        self.history_canvas.configure(
            yscrollcommand=self.history_vscroll.set,
            xscrollcommand=self.history_hscroll.set)

        # 真正放内容的 Frame
        self.history_frame = tk.Frame(self.history_canvas)
        self.history_canvas.create_window(
            (0, 0), window=self.history_frame, anchor="nw")

        # 更新 scroll region
        self.history_frame.bind(
            "<Configure>",
            lambda e: self.history_canvas.configure(
                scrollregion=self.history_canvas.bbox("all"))
        )

        # Canvas + 两条滚动条在 history_container 里排版
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_vscroll.grid(row=0, column=1, sticky="ns")
        self.history_hscroll.grid(row=1, column=0, sticky="ew")

        # 让 Canvas 单元格可伸缩
        self.history_container.grid_rowconfigure(0, weight=1)
        self.history_container.grid_columnconfigure(0, weight=1)

        # 模型相关属性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # 模型实例
        self.load_model()  # 初始化时加载模型

    def _on_mousewheel(self, event):
        """滑动鼠标滚轮 → 垂直滚动错题本面板"""
        self.history_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        """按住 Shift + 滚轮 → 水平滚动错题本面板"""
        self.history_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_images(self):
        # 获取系统缩放因子
        scaling_factor = self.root.tk.call('tk', 'scaling')
        base_size = 30
        icon_size = int(base_size * scaling_factor)  # 动态计算图标大小

        for i in range(1, MONSTER_COUNT + 1):
            # 使用PIL打开图像并缩放
            img = Image.open(f"images/{i}.png")
            width, height = img.size

            # 计算缩放比例，保持宽高比且不超过目标尺寸
            ratio = min(icon_size / width, icon_size / height)
            new_size = (int(width * ratio), int(height * ratio))

            # 高质量缩放
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为Tkinter兼容格式
            photo_img = ImageTk.PhotoImage(img_resized)
            self.images[str(i)] = photo_img

    def load_model(self):
        """初始化时加载模型"""
        try:
            if not os.path.exists('models/best_model_full.pth'):
                raise FileNotFoundError("未找到训练好的模型文件 'models/best_model_full.pth'，请先训练模型")

            try:
                model = torch.load('models/best_model_full.pth', map_location=self.device, weights_only=False)
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load('models/best_model_full.pth', map_location=self.device)
            model.eval()
            self.model = model.to(self.device)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            messagebox.showerror("严重错误", error_msg)
            self.root.destroy()  # 无法继续运行，退出程序

    def create_widgets(self):
        # 创建顶层容器
        self.top_container = tk.Frame(self.main_panel)
        self.bottom_container = tk.Frame(self.main_panel)

        # 顶部容器布局（填充整个水平空间）
        self.top_container.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.bottom_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        # 创建居中容器用于放置左右怪物框
        self.monster_center = tk.Frame(self.top_container)
        self.monster_center.pack(side=tk.TOP, anchor='center')

        # 创建左右怪物容器（添加边框和背景色）
        self.left_frame = tk.Frame(self.monster_center, borderwidth=2, relief="groove", padx=5, pady=5)
        self.right_frame = tk.Frame(self.monster_center, borderwidth=2, relief="groove", padx=5, pady=5)

        # 添加左右标题
        tk.Label(self.left_frame, text="左侧怪物", font=('Helvetica', 10, 'bold')).grid(row=0, columnspan=10)
        tk.Label(self.right_frame, text="右侧怪物", font=('Helvetica', 10, 'bold')).grid(row=0, columnspan=10)

        # 左右布局（添加显式间距并居中）
        self.left_frame.pack(side=tk.LEFT, padx=10, anchor='n', pady=5)
        self.right_frame.pack(side=tk.RIGHT, padx=10, anchor='n', pady=5)

        # 怪物输入框生成逻辑（增加行间距）
        for side, frame, monsters in [("left", self.left_frame, self.left_monsters),
                                      ("right", self.right_frame, self.right_monsters)]:
            monsters_per_row = math.ceil(MONSTER_COUNT / 4)
            for row in range(4):
                start = row * monsters_per_row + 1
                end = min((row + 1) * monsters_per_row + 1, MONSTER_COUNT + 1)
                for i in range(start, end):
                    # 图片标签增加内边距
                    tk.Label(frame, image=self.images[str(i)], padx=3, pady=3).grid(
                        row=row * 2 + 1,  # 从第1行开始
                        column=i - start,
                        sticky='ew'
                    )
                    # 输入框增加内边距
                    monsters[str(i)] = tk.Entry(frame, width=5)  # 加宽输入框
                    monsters[str(i)].grid(
                        row=row * 2 + 2,  # 下移一行
                        column=i - start,
                        pady=(0, 5)  # 底部留空
                    )

        # 结果显示区域（增加边框）
        self.result_frame = tk.Frame(self.bottom_container, relief="ridge", borderwidth=1)
        self.result_frame.pack(fill=tk.X, pady=5)

        # 使用更醒目的字体
        self.result_label = tk.Label(self.result_frame, text="Prediction: ", font=("Helvetica", 16, "bold"), fg="blue")
        self.result_label.pack(pady=3)
        self.stats_label = tk.Label(self.result_frame, text="", font=("Helvetica", 12), fg="green")
        self.stats_label.pack(pady=3)

        # 按钮区域容器（增加边框和背景）
        self.button_frame = tk.Frame(self.bottom_container, relief="groove", borderwidth=2, padx=10, pady=10)
        self.button_frame.pack(fill=tk.BOTH, expand=True)

        # 按钮布局（分左右两列布局）
        left_buttons = tk.Frame(self.button_frame)
        center_buttons = tk.Frame(self.button_frame)  # 新增中间按钮容器
        right_buttons = tk.Frame(self.button_frame)

        # 使用grid布局实现均匀分布
        left_buttons.grid(row=0, column=0, sticky='ew')
        center_buttons.grid(row=0, column=1, sticky='ew')  # 中间列
        right_buttons.grid(row=0, column=2, sticky='ew')
        self.button_frame.grid_columnconfigure((0, 1, 2), weight=1)  # 均匀分布三列

        # 左侧按钮列（控制选项）
        control_col = tk.Frame(left_buttons)
        control_col.pack(anchor='center', expand=True)

        # 时长输入组
        duration_frame = tk.Frame(control_col)
        duration_frame.pack(pady=2)
        tk.Label(duration_frame, text="训练时长:").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(duration_frame, width=6)
        self.duration_entry.insert(0, "-1")
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        # 模式选择组
        mode_frame = tk.Frame(control_col)
        mode_frame.pack(pady=2)
        self.mode_menu = tk.OptionMenu(mode_frame, self.game_mode, "单人", "30人")
        self.mode_menu.pack(side=tk.LEFT)
        self.invest_checkbox = tk.Checkbutton(mode_frame, text="投资", variable=self.is_invest)
        self.invest_checkbox.pack(side=tk.LEFT, padx=5)

        # 中间按钮列（核心操作）
        action_col = tk.Frame(center_buttons)
        action_col.pack(anchor='center', expand=True)

        # 核心操作按钮
        action_buttons = [
            ("自动获取数据", self.toggle_auto_fetch)
        ]
        # 单独处理自动获取数据按钮
        for text, cmd in action_buttons:
            btn = tk.Button(action_col, text=text, command=cmd, width=14)  # 加宽按钮
            btn.pack(pady=5, ipadx=5)
            if text == "自动获取数据":
                self.auto_fetch_button = btn
            btn.pack(pady=5, ipadx=5)

        # 创建对错按钮容器（水平排列）
        fill_buttons_frame = tk.Frame(action_col)
        fill_buttons_frame.pack(pady=2)

        # 左侧的√按钮
        tk.Button(fill_buttons_frame, text="填写√", command=self.fill_data_correct, width=8, bg="#C1E1C1").pack(
            side=tk.LEFT, padx=5)

        # 右侧的×按钮
        tk.Button(fill_buttons_frame, text="填写×", command=self.fill_data_incorrect, width=8, bg="#FFB3BA").pack(
            side=tk.RIGHT, padx=5)

        # 右侧按钮列（功能按钮）
        func_col = tk.Frame(right_buttons)
        func_col.pack(anchor='center', expand=True)

        # 预测功能组
        predict_frame = tk.Frame(func_col)
        predict_frame.pack(pady=2)
        self.predict_button = tk.Button(predict_frame, text="预测", command=self.predict, width=8, bg="#FFE4B5")
        self.predict_button.pack(side=tk.LEFT, padx=2)

        self.recognize_button = tk.Button(predict_frame, text="识别并预测", command=self.recognize, width=10,
                                          bg="#98FB98")
        self.recognize_button.pack(side=tk.LEFT, padx=2)

        self.reset_button = tk.Button(predict_frame, text="归零", command=self.reset_entries, width=6)
        self.reset_button.pack(side=tk.LEFT, padx=2)

        # 设备序列号组（独立行）
        serial_frame = tk.Frame(func_col)
        serial_frame.pack(pady=5)

        self.reselect_button = tk.Button(serial_frame, text="选择范围", command=self.reselect_roi, width=10)
        self.reselect_button.pack(side=tk.LEFT)

        tk.Label(serial_frame, text="设备号:").pack(side=tk.LEFT)
        self.serial_entry = tk.Entry(serial_frame, textvariable=self.device_serial, width=15)
        self.serial_entry.pack(side=tk.LEFT, padx=3)

        self.serial_button = tk.Button(serial_frame, text="更新", command=self.update_device_serial, width=6)
        self.serial_button.pack(side=tk.LEFT)

        # 错题本开关
        self.history_button = tk.Button(
            func_col, text="显示错题本",
            command=self.toggle_history_panel, width=10
        )
        self.history_button.pack(pady=4)  # 可以 side=tk.TOP / BOTTOM 都行

    def toggle_history_panel(self):
        if not self.history_visible:
            self.history_container.pack(side="right", fill="both", padx=5, pady=5)
            self.history_button.config(text="隐藏错题本")
            for w in self.history_frame.winfo_children():
                w.destroy()
            self.render_similar_matches(self.history_frame)
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        else:
            self.history_container.pack_forget()
            self.history_button.config(text="显示错题本")
        self.history_visible = not self.history_visible

    def load_history_data(self):
        """错题本读取的数据集，只在 __init__ 里启动时调用"""
        df = pd.read_csv(
            "arknights.csv",
            header=0,  # 默认表给有header
            engine="python",
            on_bad_lines="skip"
        )
        self.past_left = df.iloc[:, 0:56].to_numpy(float)
        self.past_right = df.iloc[:, 56:112].to_numpy(float)
        self.labels = df.iloc[:, 112].to_numpy()
        # 组合特征
        self.feat_past = np.hstack([
            self.past_left + self.past_right,
            np.abs(self.past_left - self.past_right)
        ])
        self.N_history = len(self.past_left)

    def render_similar_matches(self, parent):
        try:
            cur_left = np.zeros(56, dtype=float)
            cur_right = np.zeros(56, dtype=float)
            for name, e in self.left_monsters.items():
                v = e.get()
                if v.isdigit(): cur_left[int(name) - 1] = float(v)
            for name, e in self.right_monsters.items():
                v = e.get()
                if v.isdigit(): cur_right[int(name) - 1] = float(v)

            setL_cur = set(np.where(cur_left > 0)[0])
            setR_cur = set(np.where(cur_right > 0)[0])

            # 相似度和特征
            feat_cur = np.hstack([
                cur_left + cur_right,
                np.abs(cur_left - cur_right)
            ]).reshape(1, -1)
            sims = cosine_similarity(feat_cur, self.feat_past)[0]  # shape (N_history,)

            N = self.N_history
            # 数组
            cats = np.empty(N, np.int8)
            qdiff_other = np.empty(N, np.int16)
            match_other = np.empty(N, np.int16)
            swap = np.zeros(N, dtype=bool)

            # 分类函数
            def classify(typeL_eq, typeR_eq, cntL_eq, cntR_eq):
                if typeL_eq and typeR_eq and cntL_eq and cntR_eq:
                    return 0
                if typeL_eq and typeR_eq:
                    return 1 if (cntL_eq or cntR_eq) else 2
                if (typeL_eq and cntL_eq) or (typeR_eq and cntR_eq):
                    return 3
                if typeL_eq or typeR_eq:
                    return 4
                return 5

            # 逻辑
            for i in range(N):
                Lraw, Rraw = self.past_left[i], self.past_right[i]

                # 判断要不要镜像
                missA = len(setL_cur ^ set(np.where(Lraw > 0)[0])) + \
                        len(setR_cur ^ set(np.where(Rraw > 0)[0]))
                cntA = int(np.abs(Lraw - cur_left).sum() +
                           np.abs(Rraw - cur_right).sum())

                missB = len(setL_cur ^ set(np.where(Rraw > 0)[0])) + \
                        len(setR_cur ^ set(np.where(Lraw > 0)[0]))
                cntB = int(np.abs(Rraw - cur_left).sum() +
                           np.abs(Lraw - cur_right).sum())

                if (missB, cntB) < (missA, cntA):
                    swap[i] = True
                    Lh, Rh = Rraw, Lraw
                else:
                    Lh, Rh = Lraw, Rraw

                need_L = np.where(cur_left > 0)[0]
                need_R = np.where(cur_right > 0)[0]
                setL_h = set(np.where(Lh > 0)[0])
                setR_h = set(np.where(Rh > 0)[0])

                full_L = np.all(Lh[need_L] == cur_left[need_L])
                full_R = np.all(Rh[need_R] == cur_right[need_R])

                diff_L = int(np.abs(Lh[need_L] - cur_left[need_L]).sum())
                diff_R = int(np.abs(Rh[need_R] - cur_right[need_R]).sum())

                hit_other_R = len(setR_h & set(need_R))
                hit_other_L = len(setL_h & set(need_L))
                match_other[i] = min(hit_other_L, hit_other_R)
                if hit_other_R and not full_R:
                    qdiff_other[i] = diff_R
                elif hit_other_L and not full_L:
                    qdiff_other[i] = diff_L
                else:
                    qdiff_other[i] = 0

                # 分类
                cats[i] = classify(
                    setL_h == setL_cur,
                    setR_h == setR_cur,
                    np.array_equal(Lh, cur_left),
                    np.array_equal(Rh, cur_right)
                )

            # 排序
            order = np.lexsort((-sims, qdiff_other, -match_other, cats))
            good = order[match_other[order] > 0]
            backup = order[match_other[order] == 0]

            # 前5和前20的index
            top20_idx = np.concatenate((good, backup))[:20]
            top5_idx = top20_idx[:5]

            # 胜率计算和标题渲染
            tgtL = max((i for i, v in enumerate(cur_left) if v > 0),
                       key=cur_left.__getitem__, default=None)
            tgtR = max((i for i, v in enumerate(cur_right) if v > 0),
                       key=cur_right.__getitem__, default=None)

            lw = rw = 0
            for idx in top5_idx:
                lab = self.labels[idx]
                Lh, Rh = self.past_left[idx], self.past_right[idx]
                if swap[idx]:
                    lab = 'L' if lab == 'R' else 'R'
                    Lh, Rh = Rh, Lh
                if tgtL is not None:
                    side = 'L' if Lh[tgtL] > 0 else 'R'
                    lw += (lab == side)
                if tgtR is not None:
                    side = 'L' if Lh[tgtR] > 0 else 'R'
                    rw += (lab == side)

            # stats
            lw = 0
            for idx in top5_idx:
                lab = self.labels[idx]
                # 用本地的 swap，不要用 self._swap
                if swap[idx]:
                    lab = 'L' if lab == 'R' else 'R'
                if lab == 'L':
                    lw += 1
            rw = len(top5_idx) - lw

            # top5胜率
            n = len(top5_idx)
            if n > 0:
                left_rate = lw / n
                right_rate = rw / n
            else:
                left_rate = right_rate = 0
            input_types = set(np.where(cur_left > 0)[0]) | set(np.where(cur_right > 0)[0])
            perfect_count = 0
            for i in top5_idx:
                if swap[i]:
                    Lh, Rh = self.past_right[i], self.past_left[i]
                else:
                    Lh, Rh = self.past_left[i], self.past_right[i]
                record_types = set(np.where(Lh > 0)[0]) | set(np.where(Rh > 0)[0])
                if record_types == input_types:
                    perfect_count += 1
            # 标题
            head = tk.Frame(parent);
            head.pack(fill="x", pady=4)
            fgL, fgR = ("#E23F25", "#666") if left_rate > right_rate else ("#666", "#25ace2")
            tk.Label(head, text="近5条左右胜率：", font=("Helvetica", 12, "bold")).pack(side="left")
            tk.Label(head, text=f"左边 {left_rate:.2%}  ", fg=fgL, font=("Helvetica", 12, "bold")).pack(side="left")
            tk.Label(head, text=f"右边 {right_rate:.2%}", fg=fgR, font=("Helvetica", 12, "bold")).pack(side="left")

            # 彩蛋
            target_id_left = max((i for i, v in enumerate(cur_left) if v > 0),
                                 key=cur_left.__getitem__, default=None)
            target_id_right = max((i for i, v in enumerate(cur_right) if v > 0),
                                  key=cur_right.__getitem__, default=None)

            message = ""

            # 苏茜
            if perfect_count == len(top5_idx) and (
                    (target_id_left in (30, 35) and left_rate == 1.0) or
                    (target_id_right in (30, 35) and right_rate == 1.0)
            ):
                winner = "左边" if (target_id_left in (30, 35) and left_rate == 1.0) else "右边"
                message = f"苏茜决定全力支持{winner}！"
                msg_fg = "#ff99cc"

            # 维什戴尔
            elif perfect_count == len(top5_idx) and (left_rate == 1.0 or right_rate == 1.0):
                side = "左边" if left_rate == 1.0 else "右边"
                message = f"干员维什戴尔决定选择{side}！"
                msg_fg = "#444"

            # 刻俄柏
            elif perfect_count == len(top5_idx) and (
                    abs(left_rate - 0.20) < 1e-6 or abs(right_rate - 0.20) < 1e-6
            ):
                side = "左边" if left_rate <= 0.20 else "右边"
                message = f"干员刻俄柏觉得{side}能找到香甜的密饼！"
                msg_fg = "#444"

            # 渲染彩蛋
            if message:
                tk.Label(parent, text=message,
                         font=("Helvetica", 11, "bold"),
                         fg=msg_fg).pack(fill="x", pady=(2, 8))

            same_types = set(need_L) == set(need_R)
            has_left = np.any(cur_left > 0)
            has_right = np.any(cur_right > 0)

            # 错题本主体渲染
            self._history_parent = parent
            self._top20 = top20_idx.tolist()
            self._sims = sims
            self._swap = swap
            self._batch_idx = 0
            parent.after(0, lambda: self._render_batch(batch_size=5))

        except Exception as e:
            print("[渲染错题本失败]", e)

    def _render_batch(self, batch_size=5):
        start = self._batch_idx * batch_size
        end = start + batch_size
        parent = self._history_parent

        for rank, idx in enumerate(self._top20[start:end], start + 1):
            sims_val = self._sims[idx]
            swapped = self._swap[idx]
            Lh, Rh = (self.past_left if not swapped else self.past_right)[idx], \
                (self.past_right if not swapped else self.past_left)[idx]
            lab = self.labels[idx]
            if swapped:
                lab = 'L' if lab == 'R' else 'R'
            winL, winR = (lab == 'L'), (lab == 'R')

            # csv中的行数=局数
            real_no = idx + 2

            row = tk.Frame(parent, pady=6)
            row.pack(fill="x")

            # 局数
            tk.Label(
                row,
                text=f"第 {real_no} 局",
                font=("Helvetica", 10),
            ).pack(anchor="w", padx=4)

            # 相似度
            tk.Label(
                row,
                text=f"{rank}. 相似度 {sims_val:.2f}",
                font=("Helvetica", 10, "bold")
            ).pack(fill="x")

            # 左右阵容渲染
            for side, vec, is_win, bg_win, fg_win, bd_win in (
                    ('左', Lh, winL, "#ffe5e5", "#E23F25", "red"),
                    ('右', Rh, winR, "#e5e5ff", "#25ace2", "blue"),
            ):
                bg = bg_win if is_win else "#f0f0f0"
                fg = fg_win if is_win else "#666"
                bd = bd_win if is_win else "#aaa"
                pane = tk.Frame(
                    row,
                    bd=2,
                    relief="solid",
                    bg=bg,
                    highlightbackground=bd,
                    highlightthickness=2
                )
                pane.pack(
                    side="left",
                    expand=True,
                    fill="both",
                    padx=(8, 4) if side == '左' else (4, 8)
                )
                tk.Label(
                    pane,
                    text=f"{side}边",
                    fg=fg,
                    bg=bg,
                    font=("Helvetica", 9, "bold")
                ).pack(anchor="w", padx=4)
                inner = tk.Frame(pane, bg=bg)
                inner.pack(fill="x", padx=4, pady=2)
                for i, cnt in enumerate(vec):
                    if cnt > 0:
                        img = self.images[str(i + 1)]
                        tk.Label(inner, image=img, bg=bg).pack(side="left", padx=2)
                        tk.Label(inner, text=f"×{int(cnt)}", bg=bg) \
                            .pack(side="left", padx=(0, 6))

        self._batch_idx += 1

        # 更新滚动区域
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        if end < len(self._top20):
            parent.after(50, lambda: self._render_batch(batch_size))

    def reset_entries(self):
        for entry in self.left_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        for entry in self.right_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        self.result_label.config(text="Prediction: ")

    def fill_data_correct(self):
        result = 'R' if self.current_prediction > 0.5 else 'L'
        self.fill_data(result)
        self.total_fill_count += 1  # 更新总填写次数
        self.update_statistics()  # 更新统计信息

    def fill_data_incorrect(self):
        result = 'L' if self.current_prediction > 0.5 else 'R'
        self.fill_data(result)
        self.total_fill_count += 1  # 更新总填写次数
        self.incorrect_fill_count += 1  # 更新填写×次数
        self.update_statistics()  # 更新统计信息

    def fill_data(self, result):
        image_data = np.zeros((1, MONSTER_COUNT * 2))
        for name, entry in self.left_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) - 1] = int(value)
        for name, entry in self.right_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) + MONSTER_COUNT - 1] = int(value)
        image_data = np.append(image_data, result)
        image_data = np.nan_to_num(image_data, nan=-1)  # 替换所有NaN为-1

        # 将数据转换为列表，并添加图片名称
        data_row = image_data.tolist()

        with open('arknights.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)
        # messagebox.showinfo("Info", "Data filled successfully")

    def get_prediction(self):
        try:
            if self.model is None:
                raise RuntimeError("模型未正确初始化")

            # 准备输入数据（完全匹配ArknightsDataset的处理方式）
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            # 从界面获取数据（空值处理为0）
            for name, entry in self.left_monsters.items():
                value = entry.get()
                left_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            for name, entry in self.right_monsters.items():
                value = entry.get()
                right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            # 转换为张量并处理符号和绝对值
            left_signs = torch.sign(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            left_counts = torch.abs(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_signs = torch.sign(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_counts = torch.abs(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)

            # 预测流程
            with torch.no_grad():
                # 使用修改后的模型前向传播流程
                prediction = self.model(left_signs, left_counts, right_signs, right_counts).item()

                # 确保预测值在有效范围内
                if np.isnan(prediction) or np.isinf(prediction):
                    print("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                    prediction = 0.5

                # 检查预测结果是否在[0,1]范围内
                if prediction < 0 or prediction > 1:
                    prediction = max(0, min(1, prediction))

            return prediction
        except FileNotFoundError:
            messagebox.showerror("错误", "未找到模型文件，请先点击「训练」按钮")
            return 0.5
        except RuntimeError as e:
            if "size mismatch" in str(e):
                messagebox.showerror("错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            return 0.5
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字（0或正整数）")
            return 0.5
        except Exception as e:
            messagebox.showerror("错误", f"预测时发生错误: {str(e)}")
            return 0.5

    def predictText(self, prediction):
        # 结果解释（注意：prediction直接对应标签'R'的概率）
        right_win_prob = prediction  # 模型输出的是右方胜率
        left_win_prob = 1 - right_win_prob

        # 格式化输出
        result_text = (f"预测结果:\n"
                       f"左方胜率: {left_win_prob:.2%}\n"
                       f"右方胜率: {right_win_prob:.2%}")

        # 根据胜率设置颜色（保持与之前一致）
        self.result_label.config(text=result_text)
        if left_win_prob > 0.7:
            self.result_label.config(fg="#E23F25", font=("Helvetica", 12, "bold"))  # red
        elif left_win_prob > 0.6:
            self.result_label.config(fg="#E23F25", font=("Helvetica", 12, "bold"))
        elif right_win_prob > 0.7:
            self.result_label.config(fg="#25ace2", font=("Helvetica", 12, "bold"))  # blue
        elif right_win_prob > 0.6:
            self.result_label.config(fg="#25ace2", font=("Helvetica", 12, "bold"))
        else:
            self.result_label.config(fg="black", font=("Helvetica", 12, "bold"))

    def predict(self):
        # 保存当前预测结果用于后续数据收集
        self.current_prediction = self.get_prediction()
        self.predictText(self.current_prediction)

        if self.history_visible:
            for w in self.history_frame.winfo_children():
                w.destroy()
            self.render_similar_matches(self.history_frame)
            self.history_canvas.configure(
                scrollregion=self.history_canvas.bbox("all")
            )

    def recognize(self):
        # 如果正在进行自动获取数据，从adb加载截图
        if self.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
        else:
            screenshot = None

        if self.no_region:  # 如果尚未选择区域，从adb获取截图
            if self.first_recognize:  # 首次识别时，尝试连接adb
                self.main_roi = [
                    (int(0.2479 * loadData.screen_width), int(0.8410 * loadData.screen_height)),
                    (int(0.7526 * loadData.screen_width), int(0.9510 * loadData.screen_height))
                ]
                adb_path = loadData.adb_path  # 从loadData获取adb路径
                device_serial = loadData.device_serial  # 从loadData获取设备号
                subprocess.run(f'{adb_path} connect {device_serial}', shell=True, check=True)
                self.first_recognize = False
            screenshot = loadData.capture_screenshot()

        results = recognize.process_regions(self.main_roi, screenshot=screenshot)
        self.reset_entries()

        # 处理结果
        for res in results:
            if 'error' not in res:
                region_id = res['region_id']
                matched_id = res['matched_id']
                number = res['number']
                if matched_id != 0:
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.delete(0, tk.END)
                    entry.insert(0, number)
                    # Highlight the image if the entry already has data
                    if entry.get():
                        entry.config(bg="yellow")

        # =====================人工审核保存测试用例截图========================
        if intelligent_workers_debug & self.auto_fetch_running:  # 如果处于debug模式且处于自动模式
            # 获取截图区域
            x1 = int(0.2479 * loadData.screen_width)
            y1 = int(0.8444 * loadData.screen_height)
            x2 = int(0.7526 * loadData.screen_width)
            y2 = int(0.9491 * loadData.screen_height)
            # 截取指定区域
            roi = screenshot[y1:y2, x1:x2]

            # 处理结果
            processed_monster_ids = []  # 用于存储处理的怪物 ID
            for res in results:
                if 'error' not in res:
                    matched_id = res['matched_id']
                    if matched_id != 0:
                        processed_monster_ids.append(matched_id)  # 记录处理的怪物 ID
            # 生成唯一的文件名（使用时间戳）
            timestamp = int(time.time())
            if screenshot is not None:
                # 创建images目录（如果不存在）
                os.makedirs('data/images', exist_ok=True)
            # 将处理的怪物 ID 拼接到文件名中
            monster_ids_str = "_".join(map(str, processed_monster_ids))
            self.current_image_name = f"{timestamp}_{monster_ids_str}.png"
            self.current_image = cv2.resize(roi, (roi.shape[1] // 2, roi.shape[0] // 2))  # 保存缩放后的图片到内存
        self.predict()

    def reselect_roi(self):
        self.main_roi = recognize.select_roi()
        self.no_region = False

    def start_training(self):
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        # Update progress
        self.root.update_idletasks()

        # Simulate training process
        subprocess.run(["python", "train.py"])
        self.root.update_idletasks()

        messagebox.showinfo("Info", "Model trained successfully")

    def calculate_average_yellow(self, image):  # 检测左上角一点是否为黄色
        if image is None:
            print(f"图像加载失败")
            return None
        height, width, _ = image.shape
        # 取左上角(0,0)点
        point_color = image[0, 0]
        # 提取BGR通道值
        blue, green, red = point_color
        # 判断是否为黄色 (黄色RGB值大致为R高、G高、B低)
        is_yellow = (red > 150 and green > 150 and blue < 100)
        return is_yellow

    def save_statistics_to_log(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (f"总共填写次数: {self.total_fill_count}\n"
                      f"填写×次数: {self.incorrect_fill_count}\n"
                      f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n")
        with open("log.txt", "a") as log_file:
            log_file.write(stats_text)

    def toggle_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch_running = True
            self.auto_fetch_button.config(text="停止自动获取数据")
            self.start_time = time.time()  # 记录开始时间
            self.total_fill_count = 0  # 重置总填写次数
            self.incorrect_fill_count = 0  # 重置填写×次数
            self.update_statistics()  # 更新统计信息
            self.training_duration = float(self.duration_entry.get()) * 3600  # 获取训练时长（小时转秒）
            threading.Thread(target=self.auto_fetch_loop).start()
        else:
            self.auto_fetch_running = False
            self.auto_fetch_button.config(text="自动获取数据")
            self.update_statistics()  # 更新统计信息
            self.save_statistics_to_log()  # 保存统计信息到log.txt

    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            try:
                self.auto_fetch_data()
                self.update_statistics()  # 更新统计信息
                elapsed_time = time.time() - self.start_time
                if self.training_duration != -1 and elapsed_time >= self.training_duration:
                    self.auto_fetch_running = False
                    self.auto_fetch_button.config(text="自动获取数据")
                    self.save_statistics_to_log()  # 保存统计信息到log.txt
                    break

                # 检测一次间隔时间——————————————————————————————————
                time.sleep(0.5)
                if keyboard.is_pressed('esc'):
                    self.auto_fetch_running = False
                    self.auto_fetch_button.config(text="自动获取数据")
                    self.save_statistics_to_log()  # 保存统计信息到log.txt
                    break
            except Exception as e:
                print(f"自动获取数据出错: {str(e)}")
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                self.save_statistics_to_log()  # 保存统计信息到log.txt
                break
            # time.sleep(2)
            if keyboard.is_pressed('esc'):
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                break

    def auto_fetch_data(self):
        relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        screenshot = loadData.capture_screenshot()
        if screenshot is not None:
            results = loadData.match_images(screenshot, loadData.process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            # print("匹配结果：", results[0])
            for idx, score in results:
                if score > 0.5:
                    if idx == 0:
                        loadData.click(relative_points[0])
                        print("加入赛事")
                    elif idx == 1:
                        if self.game_mode.get() == "30人":
                            loadData.click(relative_points[1])
                            print("竞猜对决30人")
                            time.sleep(2)
                            loadData.click(relative_points[0])
                            print("开始游戏")
                        else:
                            loadData.click(relative_points[2])
                            print("自娱自乐")
                    elif idx == 2:
                        loadData.click(relative_points[0])
                        print("开始游戏")
                    elif idx in [3, 4, 5, 15]:
                        time.sleep(1)
                        # 归零
                        self.reset_entries()
                        # 识别怪物类型数量
                        self.recognize()
                        # 点击下一轮
                        if self.is_invest.get():  # 投资
                            # 根据预测结果点击投资左/右
                            if self.current_prediction > 0.5:
                                if idx == 4:
                                    loadData.click(relative_points[0])
                                else:
                                    loadData.click(relative_points[2])
                                print("投资右")
                            else:
                                if idx == 4:
                                    loadData.click(relative_points[1])
                                else:
                                    loadData.click(relative_points[3])
                                print("投资左")
                            if self.game_mode.get() == "30人":
                                time.sleep(20)  # 30人模式下，投资后需要等待20秒
                        else:  # 不投资
                            loadData.click(relative_points[4])
                            print("本轮观望")
                            time.sleep(5)

                    elif idx in [8, 9, 10, 11]:
                        # 判断本次是否填写错误
                        if self.calculate_average_yellow(screenshot):
                            self.fill_data('L')
                            if self.current_prediction > 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据左赢")
                        else:
                            self.fill_data('R')
                            if self.current_prediction < 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据右赢")
                        self.total_fill_count += 1  # 更新总填写次数
                        self.update_statistics()  # 更新统计信息
                        print("下一轮")
                        # 为填写数据操作设置冷却期
                        time.sleep(10)
                    elif idx in [6, 7, 14]:
                        print("等待战斗结束")
                    elif idx in [12, 13]:  # 返回主页
                        loadData.click(relative_points[0])
                        print("返回主页")
                    break  # 匹配到第一个结果后退出
        pass

    # 更新统计信息
    def update_statistics(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (f"总共填写次数: {self.total_fill_count} ，    "
                      f"填写×次数: {self.incorrect_fill_count}，    "
                      f"当次运行时长: {int(hours)}小时{int(minutes)}分钟")
        self.stats_label.config(text=stats_text)

    def update_device_serial(self):
        """更新设备序列号"""
        new_serial = self.device_serial.get()
        loadData.set_device_serial(new_serial)
        # 重新初始化设备连接
        loadData.device_serial = None  # 重置device_serial
        loadData.get_device_serial()  # 重新获取设备序列号
        messagebox.showinfo("提示", f"已更新模拟器序列号为: {new_serial}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ArknightsApp(root)
    root.mainloop()
