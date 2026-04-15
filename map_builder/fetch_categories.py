import os
import re
import json
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CATS = os.path.join(_HERE, "categories.json")

def find_main_js_url(session):
    """Finds the main hashed JS file from the 17173 interactive map HTML."""
    url = "https://map.17173.com/rocom/maps/shijie"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    # Looks like: <script type="module" crossorigin src="https://ue.17173cdn.com/a/terra/web/assets/index-CeHPvUtj.js"></script>
    match = re.search(r'src="(https://ue\.17173cdn\.com/a/terra/web/assets/index-[a-zA-Z0-9_]+\.js)"', r.text)
    if not match:
         match = re.search(r'src="(/a/terra/web/assets/index-[a-zA-Z0-9_]+\.js)"', r.text)
         if match:
             return "https://ue.17173cdn.com" + match.group(1)
    
    if match:
        return match.group(1)
    
    # Fallback to the latest known one if scraping fails
    return "https://ue.17173cdn.com/a/terra/web/assets/index-CeHPvUtj.js"

def extract_4010_array(js_text):
    """Finds the exactly matching 4010 map array by parsing brackets."""
    start_idx = js_text.find('4010:[{')
    if start_idx == -1:
        return None
    
    # Start parsing at the first '['
    array_start = start_idx + 4
    depth = 0
    end_idx = -1
    
    for i in range(array_start, len(js_text)):
        if js_text[i] == '[':
            depth += 1
        elif js_text[i] == ']':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
                
    if end_idx != -1:
        raw_array = js_text[array_start:end_idx+1]
        return raw_array
    return None

def clean_and_parse_js_array(js_array_str):
    """Transforms a JS object literal string into valid JSON, then parses it."""
    # This is a bit hacky, but typical for JS dicts: We add quotes around keys.
    # Matches letters/numbers before a colon, taking care not to match HTTP URLs
    # A bit risky, let's use a simpler regex that matches word boundaries.
    s = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', js_array_str)
    # Fix single quotes
    s = s.replace("'", '"')
    
    try:
        return json.loads(s)
    except Exception as e:
        print(f"[警告] JS -> JSON 转换失败，尝试更宽松的正则修复: {e}")
        # sometimes urls or other things break it, but we mostly just need title/id
        pass
    
    return None

def fallback_regex_parse(js_array_str):
    """Fallback if json.loads fails: use regex to extract the category tree directly."""
    categories_dict = {}
    
    # We split by top level objects like:
    # {game_id:1731003,title:"收集",id:1731003001,categories:[{title:"...
    
    # Break down array loosely by `categories:[`
    parts = js_array_str.split('categories:[')
    
    for i in range(1, len(parts)):
        prev_part = parts[i-1]
        parent_match = re.findall(r'title:"([^"]+)"', prev_part)
        parent_title = parent_match[-1] if parent_match else "未知"
        
        children_str = parts[i].split(']')[0]
        # find items like: {title:"眠枭之星（黄）",group_id:1731003001,id:17310030047,icon:...}
        for item in re.finditer(r'\{title:"([^"]+)",group_id:\d+,id:(\d+)', children_str):
            categories_dict[item.group(2)] = {
                "name": item.group(1),
                "group": parent_title
            }
            
    return categories_dict


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    print("正在获取 JS 主文件 URL...")
    js_url = find_main_js_url(session)
    print(f"-> {js_url}")
    
    print("下载并提取 JS 分类数据...")
    r = session.get(js_url, timeout=15)
    r.raise_for_status()
    
    raw_array = extract_4010_array(r.text)
    if not raw_array:
        print("[错误] 未在 JS 中找到地图 4010 的分类数据块！")
        return
        
    print("正在解析分类...")
    cat_dict = fallback_regex_parse(raw_array)
    
    if not cat_dict:
        print("[错误] 解析分类字典失败")
        return
        
    with open(OUTPUT_CATS, "w", encoding="utf-8") as f:
        json.dump(cat_dict, f, ensure_ascii=False, indent=2)
        
    print(f"成功！提取了 {len(cat_dict)} 种分类。")
    print(f"保存至：{OUTPUT_CATS}")
    
    # preview
    preview = list(cat_dict.items())[:3]
    for k, v in preview:
        print(f"  - {k}: {v['name']} ({v['group']})")

if __name__ == "__main__":
    main()