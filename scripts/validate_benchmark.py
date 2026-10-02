"""Check benchmark structure, citations, provenance, and recorded file hashes."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_bundle(root):
    root = Path(root)
    folder = root / 'data' / 'benchmark'
    def read_csv(name):
        with (folder / name).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    return {
        'root': root,
        'folder': folder,
        'records': read_csv('requirements.csv'),
        'review': read_csv('review.csv'),
        'source': json.loads((folder / 'source.json').read_text()),
        'clauses': json.loads((folder / 'clauses.json').read_text()),
        'manifest': json.loads((folder / 'manifest.json').read_text()),
    }


def validate(bundle, check_hashes=True):
    errors = []
    rows = bundle['records']
    reviews = bundle['review']
    originals = bundle['source']['records']
    manifest = bundle['manifest']
    references = bundle['clauses']
    lookup = {r['swe_id']: r for r in references}
    ids = [r['requirement_id'] for r in rows]
    expected_ids = [f'NASA-SR-{n:03d}' for n in range(1, 51)]
    if len(rows) != 50 or not 30 <= len(rows) <= 50:
        errors.append('Expected exactly 50 benchmark records.')
    if ids != expected_ids:
        errors.append('Requirement IDs must be unique and ordered NASA-SR-001 through NASA-SR-050.')
    if len(references) != len(lookup):
        errors.append('Duplicate SWE IDs in the clause reference.')
    if len(originals) != len(rows) or len(reviews) != len(rows):
        errors.append('Source snapshot, review log, and benchmark must contain the same number of rows.')

    for row in rows:
        name = row['requirement_id']
        for field in ['category', 'requirement_text', 'swe_id', 'npr_clause', 'verdict', 'rationale']:
            if not row.get(field, '').strip():
                errors.append(f'{name}: missing {field}.')
        if row['verdict'] not in {'Meets', 'Partial', 'Gap'}:
            errors.append(f'{name}: invalid verdict {row["verdict"]!r}.')
        clause = lookup.get(row['swe_id'])
        if not clause:
            errors.append(f'{name}: SWE ID not found in source reference.')
            continue
        if row['npr_clause'] != clause['npr_clause']:
            errors.append(f'{name}: clause does not match SWE ID.')
        if f'[{row["swe_id"]}]' not in clause['clause_text']:
            errors.append(f'{name}: source text lacks its SWE marker.')
        if not isinstance(clause['pdf_page'], int) or not 1 <= clause['pdf_page'] <= 89:
            errors.append(f'{name}: invalid PDF page locator.')
        if row['verdict'] == 'Meets' and row['gap_type']:
            errors.append(f'{name}: Meets record must not have a gap type.')
        if row['verdict'] != 'Meets' and not row['gap_type']:
            errors.append(f'{name}: non-Meets record needs a gap type.')

    for row, change, original in zip(rows, reviews, originals):
        name = row['requirement_id']
        if change['requirement_id'] != name or change['source_id'] != original['Req ID']:
            errors.append(f'{name}: source ID linkage is invalid.')
        if change['source_label'] != original['Ground Truth Label']:
            errors.append(f'{name}: original label was not preserved.')
        if change['final_verdict'] != row['verdict']:
            errors.append(f'{name}: review verdict differs from benchmark.')
        changed = row['requirement_text'] != original['Synthetic Software Requirement (NASA Project)']
        if change['text_changed'] != str(changed).lower():
            errors.append(f'{name}: text-change flag is incorrect.')
        if row['swe_id'] != original['SWE ID'] or row['npr_clause'] != str(original['NPR Clause']):
            errors.append(f'{name}: original citation was changed without an approved correction.')
        binary = 'Compliant' if row['verdict'] == 'Meets' else 'Gap'
        if binary != original['Ground Truth Label']:
            errors.append(f'{name}: final wording does not preserve the intended binary scenario.')

    counts = dict(Counter(r['verdict'] for r in rows))
    source_counts = dict(Counter(r['Ground Truth Label'] for r in originals))
    changed_count = sum(r['text_changed'] == 'true' for r in reviews)
    normalized_count = sum(r['source_id'] != r['requirement_id'] for r in reviews)
    actual = {
        'row_count': len(rows),
        'unique_swe_ids': len(lookup),
        'source_label_counts': source_counts,
        'verdict_counts': counts,
        'revised_text_count': changed_count,
        'normalized_id_count': normalized_count,
    }
    for field, value in actual.items():
        if manifest.get(field) != value:
            errors.append(f'Manifest {field} does not match the data.')

    for swe, terms in {
        'SWE-052': ['Table 1', 'hazards', 'non-conformances'],
        'SWE-027': ['a.', 'b.', 'c.', 'd.', 'e.', 'f.'],
        'SWE-087': ['a.', 'b.', 'c.', 'd.', 'e.'],
    }.items():
        clause = lookup.get(swe, {}).get('clause_text', '')
        if not all(term in clause for term in terms):
            errors.append(f'{swe}: reference omitted required table or list content.')

    if check_hashes:
        for filename, expected in manifest['files'].items():
            path = bundle['folder'] / filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'File hash mismatch: {filename}.')
        for key in ['source_pdf', 'source_html']:
            path = bundle['root'] / manifest[key]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[f'{key}_sha256']:
                errors.append(f'Standard snapshot hash mismatch: {key}.')

    if errors:
        raise ValueError('\n'.join(errors))
    return {'status': 'PASS', **actual, 'citation_matches': len(rows), 'file_hashes_checked': check_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate(load_bundle(args.root)), indent=2))


if __name__ == '__main__':
    main()
