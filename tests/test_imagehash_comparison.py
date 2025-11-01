#!/usr/bin/env python3
"""
ImageHash 알고리즘 비교 테스트
연속 60개 프레임에서 같은 프레임 3개 이상 감지 테스트
"""

import sys
import time
import base64
import io
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
import imagehash

# 서버 설정
SERVER_URL = "http://127.0.0.1:8000"
NUM_FRAMES = 60
SUCCESS_THRESHOLD = 3  # 3개 이상 같으면 성공

# 테스트할 해시 알고리즘과 파라미터
HASH_CONFIGS = [
    ("average_hash (8x8)", lambda img: imagehash.average_hash(img, hash_size=8)),
    ("average_hash (16x16)", lambda img: imagehash.average_hash(img, hash_size=16)),
    ("phash (8x8)", lambda img: imagehash.phash(img, hash_size=8)),
    ("phash (16x16)", lambda img: imagehash.phash(img, hash_size=16)),
    ("dhash (8x8)", lambda img: imagehash.dhash(img, hash_size=8)),
    ("dhash (16x16)", lambda img: imagehash.dhash(img, hash_size=16)),
    ("whash (8x8)", lambda img: imagehash.whash(img, hash_size=8)),
    ("whash (16x16)", lambda img: imagehash.whash(img, hash_size=16)),
    ("colorhash", lambda img: imagehash.colorhash(img)),
]


def collect_frames(num_frames):
    """서버에서 프레임 수집"""
    print(f"🎬 Collecting {num_frames} frames from server...")
    frames = []

    for i in range(num_frames):
        try:
            response = requests.get(f"{SERVER_URL}/state", timeout=5)
            if response.status_code == 200:
                data = response.json()

                # 프레임 추출
                visual = data.get('visual', {})
                frame_data = visual.get('screenshot_base64')
                if frame_data:
                    # base64 디코딩 (data URL 형식인 경우 처리)
                    if ',' in frame_data:
                        frame_data = frame_data.split(',')[1]

                    img_bytes = base64.b64decode(frame_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    frames.append(img)

                    print(f"  Frame {i+1}/{num_frames} collected", end='\r')
                else:
                    print(f"\n⚠️  Frame {i+1}: No frame data")
            else:
                print(f"\n⚠️  Frame {i+1}: HTTP {response.status_code}")

            # 짧은 대기 (프레임 변화를 위해)
            time.sleep(0.05)

        except Exception as e:
            print(f"\n❌ Error collecting frame {i+1}: {e}")
            continue

    print(f"\n✅ Collected {len(frames)} frames")
    return frames


def test_hash_algorithm(name, hash_func, frames):
    """특정 해시 알고리즘 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # 모든 프레임 해싱
    hashes = []
    print("  Computing hashes...", end='', flush=True)
    try:
        for frame in frames:
            h = hash_func(frame)
            hashes.append(h)
        print(" ✓")
    except Exception as e:
        print(f" ✗\n  ❌ Error: {e}")
        return None

    # 같은 해시 개수 세기
    hash_counts = defaultdict(int)
    for h in hashes:
        hash_counts[str(h)] += 1

    # 통계
    max_count = max(hash_counts.values()) if hash_counts else 0
    unique_hashes = len(hash_counts)
    duplicate_groups = sum(1 for count in hash_counts.values() if count > 1)

    # Hamming distance 분석 (연속 프레임간)
    distances = []
    for i in range(len(hashes) - 1):
        dist = hashes[i] - hashes[i+1]
        distances.append(dist)

    avg_distance = sum(distances) / len(distances) if distances else 0
    min_distance = min(distances) if distances else 0
    max_distance = max(distances) if distances else 0

    # 결과 출력
    print(f"\n  📊 Statistics:")
    print(f"    Total frames:        {len(frames)}")
    print(f"    Unique hashes:       {unique_hashes}")
    print(f"    Max duplicates:      {max_count} frames")
    print(f"    Duplicate groups:    {duplicate_groups}")
    print(f"\n  📏 Hamming Distance (consecutive frames):")
    print(f"    Average:             {avg_distance:.2f}")
    print(f"    Min:                 {min_distance}")
    print(f"    Max:                 {max_distance}")

    # 성공 판정
    success = max_count >= SUCCESS_THRESHOLD

    if success:
        print(f"\n  ✅ SUCCESS: Found {max_count} identical frames (threshold: {SUCCESS_THRESHOLD})")
    else:
        print(f"\n  ❌ FAILED: Only {max_count} identical frames (threshold: {SUCCESS_THRESHOLD})")

    # 상세 중복 정보
    duplicates = [(h, count) for h, count in hash_counts.items() if count > 1]
    if duplicates:
        print(f"\n  🔍 Duplicate hash details:")
        for h, count in sorted(duplicates, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    Hash {h[:16]}...: {count} occurrences")

    return {
        'name': name,
        'success': success,
        'unique_hashes': unique_hashes,
        'max_duplicates': max_count,
        'duplicate_groups': duplicate_groups,
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
    }


def main():
    print("="*60)
    print("ImageHash Algorithm Comparison Test")
    print("="*60)
    print(f"Server: {SERVER_URL}")
    print(f"Frames to collect: {NUM_FRAMES}")
    print(f"Success threshold: {SUCCESS_THRESHOLD} identical frames")
    print("="*60)

    # 프레임 수집
    frames = collect_frames(NUM_FRAMES)

    if len(frames) < 10:
        print(f"\n❌ Not enough frames collected ({len(frames)}). Exiting.")
        return

    print(f"\n✅ Using {len(frames)} frames for testing")

    # 각 해시 알고리즘 테스트
    results = []
    for name, hash_func in HASH_CONFIGS:
        result = test_hash_algorithm(name, hash_func, frames)
        if result:
            results.append(result)
        time.sleep(0.5)  # 짧은 대기

    # 최종 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Algorithm':<25} {'Success':<10} {'Max Dup':<10} {'Avg Dist':<10}")
    print("-"*60)

    for r in results:
        success_mark = "✅" if r['success'] else "❌"
        print(f"{r['name']:<25} {success_mark:<10} {r['max_duplicates']:<10} {r['avg_distance']:<10.2f}")

    # 권장 알고리즘
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    successful = [r for r in results if r['success']]
    if successful:
        # 성공한 것 중 avg_distance가 적당한 것 (너무 널널하지 않게)
        # avg_distance가 너무 작으면 너무 많은 걸 같다고 판단
        # avg_distance가 적당히 있으면서 max_duplicates가 높은 게 좋음

        best = max(successful, key=lambda x: (x['max_duplicates'], -x['unique_hashes']))

        print(f"\n🏆 Best performer: {best['name']}")
        print(f"   - {best['max_duplicates']} identical frames detected")
        print(f"   - {best['unique_hashes']} unique states")
        print(f"   - Average distance: {best['avg_distance']:.2f}")

        # 너무 널널하지 않은 것 추천
        moderate = [r for r in successful if r['avg_distance'] > 1.0]
        if moderate:
            print(f"\n💡 Recommended (not too loose):")
            for r in sorted(moderate, key=lambda x: x['max_duplicates'], reverse=True)[:3]:
                print(f"   - {r['name']}: {r['max_duplicates']} duplicates, avg dist {r['avg_distance']:.2f}")
    else:
        print("\n⚠️  No algorithm achieved the success threshold")
        print("   Consider lowering the threshold or adjusting parameters")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
