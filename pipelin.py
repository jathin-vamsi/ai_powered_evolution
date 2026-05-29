from pathlib import Path
import re, shutil

PIPELINE = Path.home() / "trading_bot" / "pipeline.py"
BACKUP   = Path.home() / "trading_bot" / "pipeline_pre_patch.py"

shutil.copy(PIPELINE, BACKUP)
print("Backup saved")

code = PIPELINE.read_text()

NEW_HELPERS = '''
def _autofix_code(code_str):
    code_str = code_str.replace("\\t", "    ")
    code_str = re.sub(r"```python\\s*", "", code_str)
    code_str = re.sub(r"```\\s*", "", code_str)
    code_str = re.sub(r\'("\\w+")\\s*:\\s*value\\b\', r\'\\1: 2.0\', code_str)
    code_str = code_str.replace("\\\\ ", " ")
    if "def compute_indicators" in code_str:
        already = ("c, h, l, v" in code_str or "c,h,l,v" in code_str or
                   "c = df[" in code_str or "h = df[" in code_str)
        if not already:
            lines = code_str.split("\\n")
            new_lines = []
            waiting = False
            done = False
            for line in lines:
                stripped = line.strip()
                if not done and "def compute_indicators" in line and stripped.endswith(":"):
                    waiting = True
                    new_lines.append(line)
                    continue
                if waiting and not done:
                    if stripped and not stripped.startswith("#"):
                        new_lines.append(\'    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]\')
                        done = True
                        waiting = False
                new_lines.append(line)
            code_str = "\\n".join(new_lines)
    if "def get_signals" in code_str:
        gs = re.search(r"def get_signals\\(df\\):(.*?)(?=\\ndef |\\n# EVOLVE|\\Z)", code_str, re.DOTALL)
        if gs:
            body = gs.group(1)
            uses = any(f" {v} " in body or f"({v}" in body for v in ["c", "h", "l", "v"])
            has = any(x in body for x in ["c, h, l, v", "c,h,l,v", "c = df"])
            if uses and not has:
                code_str = re.sub(
                    r"(def get_signals\\(df\\):)",
                    r\'\\1\\n    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]\',
                    code_str, count=1
                )
    if "def compute_indicators" in code_str and \'df["atr"]\' not in code_str:
        code_str = re.sub(
            r"(def compute_indicators\\(df\\):.*?)(\\n    return df)",
            r\'\\1\\n    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()\\2\',
            code_str, flags=re.DOTALL, count=1
        )
    if "def compute_indicators" in code_str and "atr_median" not in code_str:
        code_str = re.sub(
            r\'(df\\["atr"\\]\\s*=\\s*ta\\.volatility\\.AverageTrueRange[^\\n]+)\',
            r\'\\1\\n    df["atr_median"] = df["atr"].rolling(100).median()\',
            code_str, count=1
        )
    if "adx" in code_str and \'df["adx"]\' not in code_str and "def compute_indicators" in code_str:
        code_str = re.sub(
            r"(def compute_indicators\\(df\\):.*?)(\\n    return df)",
            r\'\\1\\n    _adx_i = ta.trend.ADXIndicator(h, l, c, window=14)\\n    df["adx"] = _adx_i.adx()\\2\',
            code_str, flags=re.DOTALL, count=1
        )
    if "def get_signals" in code_str and "stop_dist" not in code_str:
        code_str = re.sub(
            r"(def get_signals\\(df\\):.*?)(\\n    return df)",
            r\'\\1\\n    df["stop_dist"] = CONFIG.get("stop_atr", 2.0) * df["atr"]\\2\',
            code_str, flags=re.DOTALL, count=1
        )
    for sig in ["long_entry", "long_exit", "short_entry", "short_exit"]:
        if sig not in code_str and "def get_signals" in code_str:
            code_str = re.sub(
                r"(def get_signals\\(df\\):.*?)(\\n    return df)",
                rf\'\\1\\n    df["{sig}"] = pd.Series(False, index=df.index)\\2\',
                code_str, flags=re.DOTALL, count=1
            )
    return code_str


def _validate_code(code_str):
    import ast
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    if "def compute_indicators" not in code_str:
        return False, "Missing compute_indicators"
    if "def get_signals" not in code_str:
        return False, "Missing get_signals"
    if "long_entry" not in code_str:
        return False, "Missing long_entry signal"
    if "stop_dist" not in code_str:
        return False, "Missing stop_dist"
    return True, ""

'''

idx = code.index("def self_heal_code(broken_code, error_msg):")
code = code[:idx] + NEW_HELPERS + code[idx:]
print("Fix 1 done")

OLD_AUTOFIX = """    tmp_path = None
    # Auto-fix common errors before trying
    code_str = code_str.replace(\"\\t\", \"    \")
    import re as _re
    # Fix: name 'value' is not defined - replace bare 'value' in CONFIG
    code_str = _re.sub(r\'(\"\\\\w+\")\\\\s*:\\\\s*value\\\\b\', r\'\\\\1: 2.0\', code_str)
    # Fix: name 'c'/'h'/'l'/'v' not defined - guaranteed fix
    if \"def compute_indicators\" in code_str:
        needs_fix = ('c, h, l, v' not in code_str and
                     'c,h,l,v' not in code_str and
                     'c = df' not in code_str)
        if needs_fix:
            lines = code_str.split(\"\\n\")
            new_lines = []
            inserted = False
            for line in lines:
                new_lines.append(line)
                if not inserted and \"def compute_indicators\" in line and line.strip().endswith(\":\"):
                    new_lines.append('    c, h, l, v = df[\"close\"], df[\"high\"], df[\"low\"], df[\"volume\"]')
                    inserted = True
            code_str = \"\\n\".join(new_lines)
    # Fix: 'h' not defined separately
    if \"def compute_indicators\" in code_str and \" h,\" not in code_str and \"h = df\" not in code_str:
        import re as _re4
        code_str = _re4.sub(
            r'(def compute_indicators\\s*\\(df\\)\\s*:)',
            r'\\1\\n    c, h, l, v = df[\"close\"], df[\"high\"], df[\"low\"], df[\"volume\"]',
            code_str
        )
    # Fix: unexpected character after line continuation - remove backslash issues
    code_str = code_str.replace(\"\\\\ \", \" \").replace(\"\\\\\\\\n\", \"\\n\")
    # Fix: atr_median not defined
    if \"'atr_median'\" in code_str or \"atr_median\" not in code_str:
        if \"def compute_indicators\" in code_str and \"atr_median\" not in code_str:
            code_str = code_str.replace(
                'df[\"atr\"] = ta.volatility.AverageTrueRange',
                'df[\"atr\"] = ta.volatility.AverageTrueRange'
            )
            # Add atr_median after atr line
            import re as _re3
            code_str = _re3.sub(
                r'(df\\[\"atr\"\\]\\s*=\\s*ta\\.volatility\\.AverageTrueRange.*?\\.average_true_range\\(\\))',
                r'\\1\\n    df[\"atr_median\"] = df[\"atr\"].rolling(100).median()',
                code_str
            )
    # Fix: atr not defined - ensure atr is computed
    if \"def compute_indicators\" in code_str and 'df[\"atr\"]' not in code_str:
        # Add atr before return df in compute_indicators
        code_str = code_str.replace(
            \"    return df\\ndef get_signals\",
            '    df[\"atr\"] = ta.volatility.AverageTrueRange(df[\"high\"], df[\"low\"], df[\"close\"], window=14).average_true_range()\\n    df[\"atr_median\"] = df[\"atr\"].rolling(100).median()\\n    return df\\ndef get_signals'
        )
    # Fix: stop_dist not defined
    if \"def get_signals\" in code_str and 'stop_dist' not in code_str:
        code_str = code_str.replace(
            \"    return df\\n# EVOLVE-BLOCK-END\",
            '    df[\"stop_dist\"] = CONFIG.get(\"stop_atr\", 2.0) * df[\"atr\"]\\n    return df\\n# EVOLVE-BLOCK-END'
        )"""

NEW_AUTOFIX = """    tmp_path = None
    # Step 1: Auto-fix all known issues
    code_str = _autofix_code(code_str)
    # Step 2: Pre-validate syntax BEFORE running
    ok, err = _validate_code(code_str)
    if not ok:
        print(f\"  [{label}] pre-validation failed: {err}\")
        if \"_healed\" not in label:
            print(f\"  [{label}] self-healing...\")
            healed = self_heal_code(code_str, err)
            if healed:
                return backtest_code(healed, label+\"_healed\", phase, fast)
        return 0.0, f\"error: {err}\", []"""

if OLD_AUTOFIX in code:
    code = code.replace(OLD_AUTOFIX, NEW_AUTOFIX)
    print("Fix 2 done")
else:
    print("Fix 2 FAILED - old block not found, trying alternate method")
    # Alternate: find by line numbers
    lines = code.split("\n")
    start = None
    end = None
    for i, line in enumerate(lines):
        if "tmp_path = None" in line and start is None:
            start = i
        if start and "try:" in line and "NamedTemporaryFile" in lines[i+1] if i+1 < len(lines) else False:
            end = i
            break
    if start and end:
        new_block = NEW_AUTOFIX.split("\n")
        lines[start:end] = new_block
        code = "\n".join(lines)
        print("Fix 2 done via alternate method")
    else:
        print("Fix 2 completely failed - manual fix needed")

OLD_RULES = "RULES: df has open/high/low/close/volume, all 4 signals required as bool, stop_dist required, no vectorbt, no run_backtest, only ta functions listed above\"\"\""
NEW_RULES = """CRITICAL RULES:
1. Output ONLY the 3 sections. No extra text or markdown.
2. CONFIG values must be real numbers. NEVER use the word 'value'.
3. INDICATORS first line MUST be: c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
4. All 4 signals must end with .astype(bool)
5. SIGNALS must have: df["stop_dist"] = CONFIG["stop_atr"] * df["atr"]
6. Use ONLY the ta functions listed above.\"\"\""""

if OLD_RULES in code:
    code = code.replace(OLD_RULES, NEW_RULES)
    print("Fix 3 done")
else:
    print("Fix 3 skipped - rules line not found")

PIPELINE.write_text(code)
print("pipeline.py patched successfully!")
