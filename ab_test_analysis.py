import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
print("=" * 80)
print("A/B TEST 분석 시작")
print("=" * 80)

df = pd.read_csv('ab_experiment_user_metrics_large.csv')

print(f"\n전체 데이터 크기: {df.shape[0]:,}행 x {df.shape[1]}열")
print(f"\n데이터 미리보기:")
print(df.head())

# 2. 기본 정보 확인
print("\n" + "=" * 80)
print("그룹별 샘플 크기")
print("=" * 80)
group_counts = df['group'].value_counts()
print(group_counts)
print(f"\nControl: {group_counts['control']:,}명")
print(f"Test: {group_counts['test']:,}명")

# 3. 그룹별 주요 지표 계산
print("\n" + "=" * 80)
print("그룹별 주요 지표")
print("=" * 80)

metrics_summary = df.groupby('group').agg({
    'steps_7d': 'mean',
    'reward_points_7d': 'mean',
    'ad_revenue_7d': 'mean',
    'is_retained_d7': 'mean'
}).round(3)

metrics_summary.columns = ['평균_걸음수', '평균_리워드포인트', '평균_광고수익', 'D7_리텐션율']
print(metrics_summary)

# Control과 Test 분리
control = df[df['group'] == 'control']
test = df[df['group'] == 'test']

# 4. 통계적 검정

print("\n" + "=" * 80)
print("통계적 유의성 검정")
print("=" * 80)

results = []

# 4-1. 걸음 수 (t-test)
t_stat_steps, p_value_steps = stats.ttest_ind(test['steps_7d'], control['steps_7d'])
mean_diff_steps = test['steps_7d'].mean() - control['steps_7d'].mean()
pct_change_steps = (mean_diff_steps / control['steps_7d'].mean()) * 100

print(f"\n[1] 걸음 수 (steps_7d)")
print(f"   - Control 평균: {control['steps_7d'].mean():.2f}")
print(f"   - Test 평균: {test['steps_7d'].mean():.2f}")
print(f"   - 차이: {mean_diff_steps:.2f} ({pct_change_steps:+.2f}%)")
print(f"   - t-statistic: {t_stat_steps:.4f}")
print(f"   - p-value: {p_value_steps:.4f}")
print(f"   - 통계적 유의성: {'✓ 유의함 (p < 0.05)' if p_value_steps < 0.05 else '✗ 유의하지 않음 (p >= 0.05)'}")

results.append({
    '지표': '걸음 수 (steps_7d)',
    'Control 평균': f"{control['steps_7d'].mean():.2f}",
    'Test 평균': f"{test['steps_7d'].mean():.2f}",
    '변화율': f"{pct_change_steps:+.2f}%",
    '검정 통계량': f"{t_stat_steps:.4f}",
    'p-value': f"{p_value_steps:.4f}",
    '유의성': '유의함' if p_value_steps < 0.05 else '유의하지 않음'
})

# 4-2. 리워드 포인트 (t-test)
t_stat_reward, p_value_reward = stats.ttest_ind(test['reward_points_7d'], control['reward_points_7d'])
mean_diff_reward = test['reward_points_7d'].mean() - control['reward_points_7d'].mean()
pct_change_reward = (mean_diff_reward / control['reward_points_7d'].mean()) * 100

print(f"\n[2] 리워드 포인트 (reward_points_7d)")
print(f"   - Control 평균: {control['reward_points_7d'].mean():.2f}")
print(f"   - Test 평균: {test['reward_points_7d'].mean():.2f}")
print(f"   - 차이: {mean_diff_reward:.2f} ({pct_change_reward:+.2f}%)")
print(f"   - t-statistic: {t_stat_reward:.4f}")
print(f"   - p-value: {p_value_reward:.4f}")
print(f"   - 통계적 유의성: {'✓ 유의함 (p < 0.05)' if p_value_reward < 0.05 else '✗ 유의하지 않음 (p >= 0.05)'}")

results.append({
    '지표': '리워드 포인트 (reward_points_7d)',
    'Control 평균': f"{control['reward_points_7d'].mean():.2f}",
    'Test 평균': f"{test['reward_points_7d'].mean():.2f}",
    '변화율': f"{pct_change_reward:+.2f}%",
    '검정 통계량': f"{t_stat_reward:.4f}",
    'p-value': f"{p_value_reward:.4f}",
    '유의성': '유의함' if p_value_reward < 0.05 else '유의하지 않음'
})

# 4-3. 광고 수익 (t-test)
t_stat_revenue, p_value_revenue = stats.ttest_ind(test['ad_revenue_7d'], control['ad_revenue_7d'])
mean_diff_revenue = test['ad_revenue_7d'].mean() - control['ad_revenue_7d'].mean()
pct_change_revenue = (mean_diff_revenue / control['ad_revenue_7d'].mean()) * 100

print(f"\n[3] 광고 수익 (ad_revenue_7d)")
print(f"   - Control 평균: ${control['ad_revenue_7d'].mean():.4f}")
print(f"   - Test 평균: ${test['ad_revenue_7d'].mean():.4f}")
print(f"   - 차이: ${mean_diff_revenue:.4f} ({pct_change_revenue:+.2f}%)")
print(f"   - t-statistic: {t_stat_revenue:.4f}")
print(f"   - p-value: {p_value_revenue:.4f}")
print(f"   - 통계적 유의성: {'✓ 유의함 (p < 0.05)' if p_value_revenue < 0.05 else '✗ 유의하지 않음 (p >= 0.05)'}")

results.append({
    '지표': '광고 수익 (ad_revenue_7d)',
    'Control 평균': f"${control['ad_revenue_7d'].mean():.4f}",
    'Test 평균': f"${test['ad_revenue_7d'].mean():.4f}",
    '변화율': f"{pct_change_revenue:+.2f}%",
    '검정 통계량': f"{t_stat_revenue:.4f}",
    'p-value': f"{p_value_revenue:.4f}",
    '유의성': '유의함' if p_value_revenue < 0.05 else '유의하지 않음'
})

# 4-4. D7 리텐션 (z-test for proportions)
n_control = len(control)
n_test = len(test)
p_control = control['is_retained_d7'].mean()
p_test = test['is_retained_d7'].mean()
p_pooled = (control['is_retained_d7'].sum() + test['is_retained_d7'].sum()) / (n_control + n_test)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_test))
z_stat_retention = (p_test - p_control) / se
p_value_retention = 2 * (1 - stats.norm.cdf(abs(z_stat_retention)))
pct_change_retention = ((p_test - p_control) / p_control) * 100

print(f"\n[4] D7 리텐션 (is_retained_d7)")
print(f"   - Control 리텐션율: {p_control*100:.2f}%")
print(f"   - Test 리텐션율: {p_test*100:.2f}%")
print(f"   - 차이: {(p_test - p_control)*100:.2f}%p ({pct_change_retention:+.2f}%)")
print(f"   - z-statistic: {z_stat_retention:.4f}")
print(f"   - p-value: {p_value_retention:.4f}")
print(f"   - 통계적 유의성: {'✓ 유의함 (p < 0.05)' if p_value_retention < 0.05 else '✗ 유의하지 않음 (p >= 0.05)'}")

results.append({
    '지표': 'D7 리텐션 (is_retained_d7)',
    'Control 평균': f"{p_control*100:.2f}%",
    'Test 평균': f"{p_test*100:.2f}%",
    '변화율': f"{pct_change_retention:+.2f}%",
    '검정 통계량': f"{z_stat_retention:.4f}",
    'p-value': f"{p_value_retention:.4f}",
    '유의성': '유의함' if p_value_retention < 0.05 else '유의하지 않음'
})

# 5. 시각화
print("\n" + "=" * 80)
print("시각화 생성 중...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('A/B Test 그룹별 주요 지표 비교', fontsize=20, fontweight='bold', y=0.995)

# 5-1. 걸음 수
ax1 = axes[0, 0]
data1 = [control['steps_7d'].mean(), test['steps_7d'].mean()]
bars1 = ax1.bar(['Control', 'Test'], data1, color=['#3b82f6', '#10b981'], alpha=0.8, edgecolor='black')
ax1.set_ylabel('Average Steps', fontsize=12)
ax1.set_title(f'평균 걸음 수 (7일)\n변화율: {pct_change_steps:+.2f}% | p-value: {p_value_steps:.4f}', fontsize=13, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars1, data1)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 5-2. 리워드 포인트
ax2 = axes[0, 1]
data2 = [control['reward_points_7d'].mean(), test['reward_points_7d'].mean()]
bars2 = ax2.bar(['Control', 'Test'], data2, color=['#3b82f6', '#10b981'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Average Reward Points', fontsize=12)
ax2.set_title(f'평균 리워드 포인트 (7일)\n변화율: {pct_change_reward:+.2f}% | p-value: {p_value_reward:.4f}', fontsize=13, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars2, data2)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 5-3. 광고 수익
ax3 = axes[1, 0]
data3 = [control['ad_revenue_7d'].mean(), test['ad_revenue_7d'].mean()]
bars3 = ax3.bar(['Control', 'Test'], data3, color=['#3b82f6', '#10b981'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('Average Ad Revenue (USD)', fontsize=12)
ax3.set_title(f'평균 광고 수익 (7일)\n변화율: {pct_change_revenue:+.2f}% | p-value: {p_value_revenue:.4f}', fontsize=13, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars3, data3)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'${val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 5-4. D7 리텐션율
ax4 = axes[1, 1]
data4 = [p_control * 100, p_test * 100]
bars4 = ax4.bar(['Control', 'Test'], data4, color=['#3b82f6', '#10b981'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('D7 Retention Rate (%)', fontsize=12)
ax4.set_title(f'D7 리텐션율\n변화율: {pct_change_retention:+.2f}% | p-value: {p_value_retention:.4f}', fontsize=13, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars4, data4)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ab_test_results.png', dpi=300, bbox_inches='tight')
print("시각화 저장 완료: ab_test_results.png")

# 6. 결과를 마크다운 파일로 저장
print("\n결과를 마크다운 파일로 저장 중...")

md_content = f"""# A/B Test 분석 결과 보고서

## 1. 실험 개요

- **분석 일자**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **전체 데이터**: {df.shape[0]:,}명
- **Control 그룹**: {group_counts['control']:,}명
- **Test 그룹**: {group_counts['test']:,}명

## 2. 그룹별 주요 지표

| 지표 | Control | Test | 변화율 | 통계량 | p-value | 유의성 |
|------|---------|------|--------|--------|---------|--------|
"""

for r in results:
    md_content += f"| {r['지표']} | {r['Control 평균']} | {r['Test 평균']} | {r['변화율']} | {r['검정 통계량']} | {r['p-value']} | {r['유의성']} |\n"

md_content += f"""

## 3. 상세 분석 결과

### 3.1 평균 걸음 수 (steps_7d)
- **Control 그룹**: {control['steps_7d'].mean():.2f} steps
- **Test 그룹**: {test['steps_7d'].mean():.2f} steps
- **차이**: {mean_diff_steps:.2f} steps ({pct_change_steps:+.2f}%)
- **t-statistic**: {t_stat_steps:.4f}
- **p-value**: {p_value_steps:.4f}
- **결론**: {'통계적으로 유의한 차이가 있습니다. (p < 0.05)' if p_value_steps < 0.05 else '통계적으로 유의한 차이가 없습니다. (p >= 0.05)'}

### 3.2 평균 리워드 포인트 (reward_points_7d)
- **Control 그룹**: {control['reward_points_7d'].mean():.2f} points
- **Test 그룹**: {test['reward_points_7d'].mean():.2f} points
- **차이**: {mean_diff_reward:.2f} points ({pct_change_reward:+.2f}%)
- **t-statistic**: {t_stat_reward:.4f}
- **p-value**: {p_value_reward:.4f}
- **결론**: {'통계적으로 유의한 차이가 있습니다. (p < 0.05)' if p_value_reward < 0.05 else '통계적으로 유의한 차이가 없습니다. (p >= 0.05)'}

### 3.3 평균 광고 수익 (ad_revenue_7d)
- **Control 그룹**: ${control['ad_revenue_7d'].mean():.4f}
- **Test 그룹**: ${test['ad_revenue_7d'].mean():.4f}
- **차이**: ${mean_diff_revenue:.4f} ({pct_change_revenue:+.2f}%)
- **t-statistic**: {t_stat_revenue:.4f}
- **p-value**: {p_value_revenue:.4f}
- **결론**: {'통계적으로 유의한 차이가 있습니다. (p < 0.05)' if p_value_revenue < 0.05 else '통계적으로 유의한 차이가 없습니다. (p >= 0.05)'}

### 3.4 D7 리텐션율 (is_retained_d7)
- **Control 그룹**: {p_control*100:.2f}%
- **Test 그룹**: {p_test*100:.2f}%
- **차이**: {(p_test - p_control)*100:.2f}%p ({pct_change_retention:+.2f}%)
- **z-statistic**: {z_stat_retention:.4f}
- **p-value**: {p_value_retention:.4f}
- **결론**: {'통계적으로 유의한 차이가 있습니다. (p < 0.05)' if p_value_retention < 0.05 else '통계적으로 유의한 차이가 없습니다. (p >= 0.05)'}

## 4. 종합 결론

"""

# 유의한 지표 찾기
significant_metrics = []
if p_value_steps < 0.05:
    significant_metrics.append(f"걸음 수 ({pct_change_steps:+.2f}%)")
if p_value_reward < 0.05:
    significant_metrics.append(f"리워드 포인트 ({pct_change_reward:+.2f}%)")
if p_value_revenue < 0.05:
    significant_metrics.append(f"광고 수익 ({pct_change_revenue:+.2f}%)")
if p_value_retention < 0.05:
    significant_metrics.append(f"D7 리텐션 ({pct_change_retention:+.2f}%)")

if significant_metrics:
    md_content += f"이번 A/B 테스트에서 다음 지표들이 통계적으로 유의한 차이를 보였습니다:\n\n"
    for metric in significant_metrics:
        md_content += f"- {metric}\n"
else:
    md_content += "이번 A/B 테스트에서 통계적으로 유의한 차이를 보인 지표가 없습니다.\n"

md_content += f"""

## 5. 권장사항

분석 결과를 바탕으로 다음과 같은 액션을 권장합니다:

1. **유의한 개선이 있는 경우**: Test 버전을 전체 사용자에게 배포
2. **유의한 차이가 없는 경우**: 추가 실험 또는 다른 접근법 고려
3. **부정적 영향이 있는 경우**: Test 버전 중단 및 원인 분석

---
*본 보고서는 자동으로 생성되었습니다.*
"""

with open('ab_test_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print("마크다운 보고서 저장 완료: ab_test_analysis_report.md")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n생성된 파일:")
print(f"1. ab_test_results.png - 시각화 차트")
print(f"2. ab_test_analysis_report.md - 분석 보고서")
