import json
from collections import Counter, defaultdict
from datetime import datetime

# Load knowledge file
with open('.pokeagent_cache/knowledge.json', 'r') as f:
    data = json.load(f)

entries = data['entries']

print("=" * 80)
print("KNOWLEDGE BASE 분석")
print("=" * 80)
print()

# Basic stats
print(f"📊 총 지식 항목 수: {len(entries)}")
print()

# Count by milestone
milestones = Counter(e['created_milestone'] for e in entries)
print("🎯 Milestone별 학습된 지식:")
for milestone, count in milestones.most_common():
    print(f"  - {milestone}: {count}개")
print()

# Recent entries
recent_entries = sorted(entries, key=lambda x: x['created_step'], reverse=True)[:10]
print("🕐 최근 10개 지식 (step 순):")
for e in recent_entries:
    print(f"  Step {e['created_step']}: {e['content'][:80]}...")
print()

# Find updated entries
updated = [e for e in entries if e['updated_step'] is not None]
print(f"🔄 업데이트된 지식: {len(updated)}개 / {len(entries)}개 ({len(updated)/len(entries)*100:.1f}%)")
print()

# Check for duplicate IDs (potential issue)
id_counts = Counter(e['id'] for e in entries)
duplicates = {id_: count for id_, count in id_counts.items() if count > 1}
if duplicates:
    print(f"⚠️  중복된 ID 발견: {len(duplicates)}개")
    for id_, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {id_}: {count}번 중복")
    print()

# Analyze location mentions in recent entries
print("📍 최근 100개 지식에서 언급된 위치:")
recent_100 = sorted(entries, key=lambda x: x['created_step'], reverse=True)[:100]
location_keywords = defaultdict(int)
for e in recent_100:
    content = e['content'].lower()
    if 'oldale' in content:
        location_keywords['Oldale Town'] += 1
    if 'route 101' in content:
        location_keywords['Route 101'] += 1
    if 'route 102' in content:
        location_keywords['Route 102'] += 1
    if 'littleroot' in content:
        location_keywords['Littleroot'] += 1

for loc, count in sorted(location_keywords.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {loc}: {count}회")
print()

# Analyze coordinate mentions in recent entries
import re
coord_pattern = r'\((\d+),(\d+)\)'
recent_coords = []
for e in recent_100:
    matches = re.findall(coord_pattern, e['content'])
    for match in matches:
        recent_coords.append((int(match[0]), int(match[1])))

if recent_coords:
    coord_counts = Counter(recent_coords)
    print("📌 최근 100개 지식에서 가장 많이 언급된 좌표:")
    for coord, count in coord_counts.most_common(10):
        print(f"  - {coord}: {count}회")
    print()

# Check last milestone
last_entry = sorted(entries, key=lambda x: x['created_step'], reverse=True)[0]
print(f"🎮 현재 목표 milestone: {last_entry['created_milestone']}")
print(f"🔢 최근 step 번호: {last_entry['created_step']}")
print()

# Count entries in last 100 steps
last_step = last_entry['created_step']
recent_100_steps = [e for e in entries if e['created_step'] > last_step - 100]
print(f"📈 최근 100 step에서 학습한 지식: {len(recent_100_steps)}개")
if len(recent_100_steps) > 50:
    print("   ⚠️  높은 학습률 - agent가 같은 문제에 반복적으로 막혀있을 가능성")
print()
