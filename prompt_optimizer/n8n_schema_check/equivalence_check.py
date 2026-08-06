"""Assert the Python port produces IDENTICAL findings to check_params.js."""
import json, subprocess, sys, os
sys.path.insert(0, '/Users/ryan.lung/Documents/n8n')
from prompt_optimizer.schema_check import check_workflow

JS = '/Users/ryan.lung/Documents/n8n/prompt_optimizer/n8n_schema_check/check_params.js'
CWD = os.path.dirname(JS)

def run_js(wf):
    p = subprocess.run(['node', JS], input=json.dumps(wf), capture_output=True, text=True, cwd=CWD)
    return json.loads(p.stdout)

def norm(d):
    """Compare only the finding categories, ignoring warnings ordering."""
    return {k: d.get(k, []) for k in
            ('issues','invalidValues','danglingNodeReferences','unknownNodeTypes','unknownTypeVersions')}

def compare(name, wf):
    js, py = run_js(wf), check_workflow(wf)
    if 'setupError' in js:
        print(f"  SKIP {name}: js setupError"); return None
    a, b = norm(js), norm(py)
    if a == b:
        print(f"  MATCH  {name}")
        return True
    print(f"  DIFF   {name}")
    for k in a:
        if a[k] != b[k]:
            print(f"     [{k}]\n       js: {json.dumps(a[k])[:400]}\n       py: {json.dumps(b[k])[:400]}")
    return False
