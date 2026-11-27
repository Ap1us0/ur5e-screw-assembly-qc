import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from .detector import LABELS, SITES
from .workflow import VERDICTS, inspect_workpiece


FAULT_OPTIONS = {'正常': 'none', '主视角反光': 'glare', '主视角模糊': 'blur', '两个视角均不可用': 'both_bad'}
STATE_COLOR = {'pass': '#236653', 'present': '#236653', 'missing': '#a1372d', 'review': '#8a5515'}


def open_panel(detector, source, factory, output):
    root = tk.Tk()
    root.title('UR5e 螺丝质检')
    root.minsize(745, 455)
    root.option_add('*Font', ('Microsoft YaHei UI', 10))
    shell = ttk.Frame(root, padding=22)
    shell.pack(fill='both', expand=True)
    ttk.Label(shell, text='螺丝质检', font=('Microsoft YaHei UI', 24, 'bold')).pack(anchor='w')
    ttk.Label(shell, text='UR5e · ' + ('MuJoCo 仿真' if source == 'simulation' else '实机工站')).pack(anchor='w')

    missing = {site: tk.BooleanVar(root) for site in SITES}
    fault = tk.StringVar(root, '正常')
    fault_site = tk.StringVar(root, 'S2')
    controls = []
    if source == 'simulation':
        setup = ttk.LabelFrame(shell, text='仿真工况', padding=10)
        setup.pack(fill='x', pady=(15, 0))
        ttk.Label(setup, text='缺件').pack(side='left', padx=(0, 8))
        for site, checked in missing.items():
            item = ttk.Checkbutton(setup, text=site, variable=checked)
            item.pack(side='left', padx=(0, 6))
            controls.append(item)
        ttk.Label(setup, text='图像').pack(side='left', padx=(18, 8))
        for var, values, width in ((fault, tuple(FAULT_OPTIONS), 17), (fault_site, SITES, 5)):
            item = ttk.Combobox(setup, textvariable=var, values=values, state='readonly', width=width)
            item.pack(side='left', padx=(0, 6))
            controls.append(item)

    bar = ttk.Frame(shell)
    bar.pack(fill='x', pady=14)
    start_button = ttk.Button(bar, text='开始质检')
    start_button.pack(side='left')
    progress = ttk.Label(bar, text='等待启动')
    progress.pack(side='left', padx=14)
    report_button = ttk.Button(bar, text='查看照片与报告', state='disabled')
    report_button.pack(side='right')
    verdict = ttk.Label(shell, text='整件判定  —', font=('Microsoft YaHei UI', 15, 'bold'))
    verdict.pack(anchor='w', pady=(0, 10))

    cards = ttk.Frame(shell)
    cards.pack(fill='both', expand=True)
    labels = {}
    photos = {}
    for index, site in enumerate(SITES):
        cards.columnconfigure(index, weight=1, uniform='position')
        card = ttk.LabelFrame(cards, text=site, padding=8)
        card.grid(row=0, column=index, sticky='nsew', padx=(0, 8) if index < 2 else 0)
        state = ttk.Label(card, text='未拍摄')
        state.pack(anchor='w')
        box = ttk.Label(card)
        box.pack(pady=(8, 0))
        labels[site] = (state, box)

    events = queue.Queue()
    run_path = None
    active = False

    def launch():
        nonlocal active, run_path
        if active:
            return
        active = True
        run_path = None
        start_button.configure(state='disabled')
        report_button.configure(state='disabled')
        for control in controls:
            control.state(['disabled'])
        for state, box in labels.values():
            state.configure(text='未拍摄', foreground='')
            box.configure(image='')
        photos.clear()
        verdict.configure(text='整件判定  —', foreground='')
        progress.configure(text='正在启动')
        absent = tuple(site for site, checked in missing.items() if checked.get())
        effect, position = FAULT_OPTIONS[fault.get()], fault_site.get()

        def run():
            def open_rig():
                return factory(absent, effect, position) if source == 'simulation' else factory()

            try:
                result = inspect_workpiece(detector, source, open_rig, output,
                                           lambda message: events.put(('progress', message)))
                events.put(('result', result))
            except Exception as exc:
                events.put(('error', str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def open_report():
        if run_path:
            os.startfile(str(run_path / 'report.html'))

    def update_panel():
        nonlocal active, run_path
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                progress.configure(text=payload)
                continue
            active = False
            start_button.configure(state='normal')
            for control in controls:
                control.state(['!disabled'])
            if kind == 'error':
                verdict.configure(text='流程异常', foreground=STATE_COLOR['review'])
                progress.configure(text=payload)
                continue
            run_path, report = payload
            report_button.configure(state='normal')
            progress.configure(text='已完成' if not report['error'] else '设备流程异常，详见报告')
            outcome = report['verdict']
            verdict.configure(text=f'整件判定  {VERDICTS[outcome]}', foreground=STATE_COLOR[outcome])
            for record in report['positions']:
                site, state = record['site'], record['state']
                label, box = labels[site]
                retry = ' · 已补拍' if len(record['attempts']) == 2 else ''
                label.configure(text=LABELS[state] + retry, foreground=STATE_COLOR[state])
                if record['photo']:
                    image = tk.PhotoImage(file=str(Path(run_path) / record['photo']))
                    factor = max(1, (image.width() + 219) // 220, (image.height() + 159) // 160)
                    photos[site] = image.subsample(factor)
                    box.configure(image=photos[site])
        root.after(60, update_panel)

    def close():
        if active:
            progress.configure(text='本轮运行结束后可关闭窗口')
        else:
            root.destroy()

    start_button.configure(command=launch)
    report_button.configure(command=open_report)
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(60, update_panel)
    root.mainloop()
