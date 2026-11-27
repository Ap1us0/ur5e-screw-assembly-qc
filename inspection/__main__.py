import argparse
import sys
from pathlib import Path

from .acceptance import evaluate, parse_missing, simulate_acceptance
from .detector import Detector, SITES
from .samples import capture_samples, generate_sim_samples, verify_samples, write_csv
from .workflow import VERDICTS, inspect_workpiece, new_run


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / 'samples' / 'simulation'
OUTPUT = ROOT / 'outputs'


def make_factory(station, detector=None):
    if station:
        from .hardware import Hardware, StationConfig

        config = StationConfig(station)
        if detector and detector.frame_size != config.frame_size:
            raise ValueError('相机与检测配置的分辨率不同')
        return 'hardware', lambda: Hardware(config)

    def open_simulation(missing=(), fault='none', fault_site='S2', seed=0):
        from .simulation import Simulation

        rig = Simulation()
        rig.reset(missing, fault, fault_site, seed)
        return rig

    return 'simulation', open_simulation


def add_run_options(command):
    command.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    command.add_argument('--station', type=Path, help='启用真实 UR5e 与 UVC 相机')
    command.add_argument('--output', type=Path, default=OUTPUT / 'runs')


def main():
    parser = argparse.ArgumentParser(description='UR5e 固定夹具螺丝质检')
    commands = parser.add_subparsers(dest='action', required=True)
    gui = commands.add_parser('gui', help='启动操作台')
    add_run_options(gui)
    run = commands.add_parser('run', help='运行单轮巡检')
    add_run_options(run)
    run.add_argument('--missing', default='', help='仿真缺件位，例如 S1,S3')
    run.add_argument('--fault', choices=('none', 'glare', 'blur', 'both_bad'), default='none')
    run.add_argument('--fault-site', choices=SITES, default='S2')
    run.add_argument('--seed', type=int, default=0, help='仿真扰动种子')
    run.add_argument('--viewer', action='store_true', help='在 MuJoCo Viewer 中按 Enter 开始仿真')
    run.add_argument('--video', action='store_true', help='录制本轮 MuJoCo 仿真并保存相机和关节轨迹')
    assembly = commands.add_parser('assemble', help='依次装配三个螺丝位，随后统一质检')
    assembly.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    assembly.add_argument('--output', type=Path, default=OUTPUT / 'assembly')
    assembly.add_argument('--viewer', action='store_true', help='用 MuJoCo Viewer 启动并观看装配')
    assembly.add_argument('--video', action='store_true', help='保存运动及局部装配录像')
    assembly.add_argument('--fault', choices=('none', 'misaligned', 'blocked', 'retryable_jam', 'cross_thread',
                                              'damaged_thread', 'stripped_thread', 'empty_feeder'),
                          default='none')
    assembly.add_argument('--site', choices=('all', *SITES), default='all', help='默认装配三处，也可单独验证一个位置')
    assembly.add_argument('--fault-site', choices=SITES, help='故障注入位置，三处模式默认 S2')
    assembly.add_argument('--seed', type=int, default=0, help='非零种子启用供料、对孔、摩擦和成像扰动')
    assembly.add_argument('--image-fault', choices=('none', 'glare', 'blur', 'both_bad'), default='none')
    assembly.add_argument('--image-site', choices=SITES, default='S2')
    assembly.add_argument('--motion', choices=('optimized', 'baseline'), default='optimized')
    assembly.add_argument('--alignment', choices=('fixed', 'vision'), default='fixed', help='定位方式')
    assembly.add_argument('--fixture-shift', nargs=3, type=float, metavar=('X_MM', 'Y_MM', 'YAW_DEG'),
                          default=(0, 0, 0), help='仿真中实际夹具的平面平移和转角')
    assembly.add_argument('--alignment-fault', choices=('none', 'glare', 'occluded'), default='none')
    assembly.add_argument('--alignment-fault-site', choices=SITES, default='S2')
    assembly.add_argument('--insertion', choices=('force', 'fixed'), default='force', help='力反馈找正或固定入孔对照')
    assembly.add_argument('--entry-offset', nargs=2, type=float, metavar=('X_MM', 'Y_MM'), default=(0, 0),
                          help='在装配目标中注入横向偏差，不改变孔位和相机')
    assembly.add_argument('--entry-offset-site', choices=SITES, help='偏差位置，默认与故障位置相同')
    assembly.add_argument('--entry-tilt', nargs=2, type=float, metavar=('RX_DEG', 'RY_DEG'),
                          help='斜咬牙工况的电批入牙倾角')
    assembly.add_argument('--recovery', choices=('stop', 'retry', 'realign'), default='realign',
                          help='卡滞后停转、原姿态重试或重新找正后重试')
    assembly.add_argument('--strip-depth', type=float, metavar='MM', help='滑牙工况中承载牙段结束的深度')
    build = commands.add_parser('build-station', help='从真实示教点生成 PolyScope 5 脚本')
    build.add_argument('--station', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    collect = commands.add_parser('collect', help='从真实工站采集六个视角的标注样图')
    collect.add_argument('--station', type=Path, required=True)
    collect.add_argument('--missing', required=True, help='人工标注缺件位，完好件填空字符串')
    collect.add_argument('--output', type=Path, default=OUTPUT / 'capture')
    samples = commands.add_parser('verify-samples', help='从清单离线验证有件、缺件和坏图')
    samples.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    samples.add_argument('--manifest', type=Path, default=SAMPLES / 'holdout' / 'samples.csv')
    samples.add_argument('--output', type=Path, default=OUTPUT / 'sample-verification.json')
    generate = commands.add_parser('generate-sim-samples', help='从 MuJoCo 重新生成样图库')
    generate.add_argument('--output', type=Path, default=SAMPLES)
    sim = commands.add_parser('simulate-acceptance', help='运行仿真 20 轮验收')
    sim.add_argument('--output', type=Path, default=OUTPUT / 'simulation-acceptance')
    batch = commands.add_parser('assembly-acceptance', help='运行装配与质检闭环 20 轮验收')
    batch.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    batch.add_argument('--output', type=Path, default=OUTPUT / 'assembly-acceptance')
    batch.add_argument('--jobs', type=int, default=1, help='并发仿真进程数')
    batch.add_argument('--motion', choices=('optimized', 'baseline'), default='optimized')
    compare = commands.add_parser('compare-motion', help='对比两批相同工况的仿真节拍及判定')
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument('--optimized', type=Path, required=True)
    compare.add_argument('--output', type=Path, default=OUTPUT / 'motion-comparison')
    align_study = commands.add_parser('alignment-study', help='对比固定示教和腕部视觉定位在夹具偏移下的效果')
    align_study.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    align_study.add_argument('--output', type=Path, default=OUTPUT / 'vision-alignment' / 'study')
    align_study.add_argument('--jobs', type=int, default=1)
    force_study = commands.add_parser('force-study', help='对比固定入孔与力反馈在偏孔和卡滞下的表现')
    force_study.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    force_study.add_argument('--output', type=Path, default=OUTPUT / 'force-insertion' / 'study')
    force_study.add_argument('--jobs', type=int, default=1)
    thread_study = commands.add_parser('thread-study', help='对比斜咬牙后停转、原姿态重试和扶正重试')
    thread_study.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    thread_study.add_argument('--output', type=Path, default=OUTPUT / 'cross-thread' / 'study')
    thread_study.add_argument('--jobs', type=int, default=1)
    health_study = commands.add_parser('thread-health-study', help='核对正常、斜咬牙、牙侧损伤及滑牙的判定')
    health_study.add_argument('--profile', type=Path, default=SAMPLES / 'profile.json')
    health_study.add_argument('--output', type=Path, default=OUTPUT / 'thread-health' / 'study')
    health_study.add_argument('--jobs', type=int, default=1)
    accept = commands.add_parser('evaluate', help='根据独立真值记录核对 20 轮报告')
    accept.add_argument('--manifest', type=Path, required=True)
    accept.add_argument('--source', choices=('hardware', 'simulation'), default='hardware')
    accept.add_argument('--output', type=Path, default=OUTPUT / 'acceptance.json')
    args = parser.parse_args()

    try:
        if args.action in ('gui', 'run'):
            detector = Detector(args.profile)
            source, factory = make_factory(args.station, detector)
            if source != detector.source:
                raise ValueError(f'检测配置来源 {detector.source} 与设备模式 {source} 不符')
            if args.action == 'gui':
                from .panel import open_panel

                open_panel(detector, source, factory, args.output)
                return 0
            missing = parse_missing(args.missing)
            if source == 'hardware' and (missing or args.fault != 'none' or args.seed or args.fault_site != 'S2'
                                         or args.viewer or args.video):
                raise ValueError('实机巡检不接受仿真工况参数')
            if args.viewer or args.video:
                from .demo import run_demo

                outcome = run_demo(detector, args.output, missing, args.fault, args.fault_site, args.seed,
                                   viewer=args.viewer, video=args.video)
                if outcome is None:
                    return 0
                _, report = outcome
            else:
                open_rig = (lambda: factory(missing, args.fault, args.fault_site, args.seed)) if source == 'simulation' else factory
                folder, report = inspect_workpiece(detector, source, open_rig, args.output, print)
                print(f'整件 {VERDICTS[report["verdict"]]} · {folder / "report.html"}')
            if report['error']:
                print(report['error'], file=sys.stderr)
                return 1
        elif args.action == 'assemble':
            from .assembly_run import run_assembly
            from .vision_alignment import FixtureShift

            sites = SITES if args.site == 'all' else (args.site,)
            fault_site = args.fault_site or ('S2' if args.site == 'all' else args.site)
            result = run_assembly(Detector(args.profile), args.output, args.fault, sites=sites, fault_site=fault_site,
                                  seed=args.seed, image_fault=args.image_fault, image_site=args.image_site, motion=args.motion,
                                  alignment=args.alignment, fixture_shift=FixtureShift(*args.fixture_shift),
                                  alignment_fault=args.alignment_fault, alignment_fault_site=args.alignment_fault_site,
                                  insertion=args.insertion, entry_offset_mm=args.entry_offset,
                                  entry_offset_site=args.entry_offset_site or fault_site,
                                  entry_tilt_deg=args.entry_tilt, recovery=args.recovery,
                                  strip_depth_mm=args.strip_depth,
                                  viewer=args.viewer, video=args.video)
            if result is not None:
                return 0 if result[1]['status'] == 'pass' else 1
        elif args.action == 'build-station':
            from .urscript import build_script

            build_script(args.station, args.output)
            print(args.output.resolve())
        elif args.action == 'collect':
            missing = parse_missing(args.missing)
            _, factory = make_factory(args.station)
            folder = new_run(args.output)
            print(f'等待已启用的 UR5e 工站 · {folder}', flush=True)
            rig = factory()
            try:
                rows = capture_samples(rig, folder, missing)
            finally:
                rig.close()
            write_csv(folder / 'samples.csv', ('site', 'angle', 'expected', 'image'), rows)
            print(folder / 'samples.csv')
        elif args.action == 'verify-samples':
            detector = Detector(args.profile)
            if detector.source == 'hardware' and args.manifest.resolve() == SAMPLES / 'holdout' / 'samples.csv':
                raise ValueError('实机配置请用 --manifest 指定独立的实物样图清单')
            result = verify_samples(detector, args.manifest, args.output)
            print(f'样图 {result["correct"]}/{result["total"]} · 缺件误判有件 {result["false_pass"]} · {args.output}')
            return 0 if result['passed'] else 1
        elif args.action == 'generate-sim-samples':
            generate_sim_samples(args.output)
            print(args.output.resolve())
        elif args.action == 'assembly-acceptance':
            from .assembly_acceptance import run_acceptance

            folder, result = run_acceptance(args.profile.resolve(), args.output.resolve(), args.jobs, args.motion)
            print(f'装配闭环 {result["correct"]}/{result["rounds"]} · 误放行 {result["false_pass"]} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action == 'compare-motion':
            from .motion_comparison import compare

            folder, result = compare(args.baseline, args.optimized, args.output)
            print(f'节拍对比{"通过" if result["passed"] else "未通过"} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action == 'alignment-study':
            from .alignment_study import run_study

            folder, result = run_study(args.profile.resolve(), args.output.resolve(), args.jobs)
            print(f'视觉定位对比 {result["correct"]}/{result["rounds"]} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action == 'force-study':
            from .force_study import run_study

            folder, result = run_study(args.profile.resolve(), args.output.resolve(), args.jobs)
            print(f'力反馈对比 {result["correct"]}/{result["rounds"]} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action == 'thread-study':
            from .thread_study import run_thread_study

            folder, result = run_thread_study(args.profile.resolve(), args.output.resolve(), args.jobs)
            print(f'斜咬牙恢复对比 {result["correct"]}/{result["rounds"]} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action == 'thread-health-study':
            from .thread_health_study import run_health_study

            folder, result = run_health_study(args.profile.resolve(), args.output.resolve(), args.jobs)
            print(f'牙损判定 {result["correct"]}/{result["rounds"]} · 坏牙误放行 {result["false_pass"]} · {folder / "report.html"}')
            return 0 if result['passed'] else 1
        elif args.action in ('evaluate', 'simulate-acceptance'):
            if args.action == 'evaluate':
                result = evaluate(args.manifest, args.output, args.source)
                path = args.output
            else:
                folder, result = simulate_acceptance(Detector(SAMPLES / 'profile.json'), args.output)
                path = folder / 'acceptance.json'
            print(f'{result["source"]} {result["correct"]}/20 · 缺件误放行 {result["false_pass"]} · {path}')
            return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
