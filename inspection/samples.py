import csv
import json
from pathlib import Path

from .detector import ANGLES, MISSING, PRESENT, REVIEW, SITES, load_image, save_image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture_samples(rig, root, missing):
    rows = []
    for site in SITES:
        label = MISSING if site in missing else PRESENT
        for angle in ANGLES:
            image = rig.capture(site, angle)
            filename = f'{site}_{angle}_{label}.png'
            save_image(root / filename, image)
            rows.append(dict(site=site, angle=angle, expected=label, image=filename))
    rig.finish()
    return rows


def verify_samples(detector, manifest, output):
    manifest = Path(manifest)
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('样图清单为空')
    results = []
    classes = dict.fromkeys((PRESENT, MISSING, REVIEW), 0)
    coverage = set()
    for row in rows:
        site, angle, expected = row['site'], row['angle'], row['expected']
        if site not in SITES or angle not in ANGLES or expected not in classes:
            raise ValueError(f'样图清单项目无效 {row}')
        classes[expected] += 1
        coverage.add((site, angle, expected))
        image = load_image(manifest.parent / row['image'])
        observation = detector.inspect(site, angle, image)
        results.append({**row, **observation.record(), 'correct': expected == observation.state})
    required = {(site, angle, label) for site in SITES for angle in ANGLES for label in (PRESENT, MISSING)}
    correct = sum(row['correct'] for row in results)
    false_pass = sum(row['expected'] == MISSING and row['state'] == PRESENT for row in results)
    report = dict(source=detector.source, total=len(rows), correct=correct, false_pass=false_pass, classes=classes,
                  uncovered=sorted(required - coverage), passed=coverage >= required and correct == len(rows),
                  samples=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def generate_sim_samples(root):
    from .simulation import Simulation

    root.mkdir(parents=True, exist_ok=True)
    rig = Simulation()
    profile = dict(version=1, source=rig.source, frame_size=[640, 480], views={site: {} for site in SITES})
    reference = root / 'references'
    reference.mkdir(exist_ok=True)
    holdout = root / 'holdout'
    holdout.mkdir(exist_ok=True)
    try:
        for label, missing in ((PRESENT, ()), (MISSING, SITES)):
            rig.reset(missing)
            for row in capture_samples(rig, reference, missing):
                site, angle = row['site'], row['angle']
                spec = profile['views'][site].setdefault(angle, dict(roi=[269, 189, 102, 102], shift_px=6,
                    min_sharpness=45, max_glare=0.12, max_dark=0.18, min_score=0.68, min_margin=0.20))
                spec[label] = ['references/' + row['image']]
        rows = []
        for seed, label, missing in ((41, PRESENT, ()), (53, MISSING, SITES)):
            rig.reset(missing, seed=seed)
            rows.extend(capture_samples(rig, holdout, missing))
        for fault in ('glare', 'blur'):
            rig.reset(fault=fault, seed=63)
            filename = f'S2_primary_{fault}.png'
            save_image(holdout / filename, rig.capture('S2', 'primary'))
            rows.append(dict(site='S2', angle='primary', expected=REVIEW, image=filename))
        write_csv(holdout / 'samples.csv', ('site', 'angle', 'expected', 'image'), rows)
        write_json(root / 'profile.json', profile)
    finally:
        rig.close()
