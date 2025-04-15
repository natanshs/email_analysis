import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
email_table = pd.read_csv(r'data\email_table.csv')
email_opened_table = pd.read_csv(r'data\email_opened_table.csv')
link_clicked_table = pd.read_csv(r'data\link_clicked_table (1).csv')

print("Data loaded successfully!")

# Check the shape of dataframes
print(f"Total emails: {len(email_table)}")
print(f"Emails opened: {len(email_opened_table)}")
print(f"Emails clicked: {len(link_clicked_table)}")

# Add flags for opened and clicked emails
email_table['opened'] = email_table['email_id'].isin(email_opened_table['email_id']).astype(int)
email_table['clicked'] = email_table['email_id'].isin(link_clicked_table['email_id']).astype(int)

# Basic data exploration
print("\nEmail table columns:", email_table.columns.tolist())
print("\nEmail table sample:")
print(email_table.head())

# Answer Question 1: What percentage of users opened the email and clicked the link?
total_emails = len(email_table)
opened_emails = sum(email_table['opened'])
clicked_emails = sum(email_table['clicked'])

open_rate = opened_emails / total_emails * 100
click_rate = clicked_emails / total_emails * 100
ctr_of_opened = clicked_emails / opened_emails * 100 if opened_emails > 0 else 0

print("\n----- Question 1: Email Campaign Metrics -----")
print(f"Total emails sent: {total_emails}")
print(f"Emails opened: {opened_emails} ({open_rate:.2f}%)")
print(f"Emails clicked: {clicked_emails} ({click_rate:.2f}%)")
print(f"CTR of opened emails (clicks/opens): {ctr_of_opened:.2f}%")

# Answer Question 2: Build a predictive model to optimize email sends
print("\n----- Question 2: Predictive Model for Email Optimization -----")

# Prepare features and target
X = email_table.drop(['email_id', 'opened', 'clicked'], axis=1)
y = email_table['clicked']

# Print feature information
print("\nFeatures used for prediction:")
for col in X.columns:
    print(f"- {col}: {X[col].nunique()} unique values")
    
# One-hot encode categorical variables
categorical_features = ['email_text', 'email_version', 'weekday', 'user_country']
numeric_features = ['hour', 'user_past_purchases']

# Create a column transformer to handle different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nModel Performance (Cross-validation ROC-AUC): {np.mean(cv_scores):.4f}")

# Get feature importance
def get_feature_importance(model):
    if not hasattr(model['classifier'], 'feature_importances_'):
        return None
    
    # Get feature names after preprocessing
    ohe = model['preprocessor'].transformers_[0][1]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = cat_feature_names + numeric_features
    
    # Get feature importances
    importances = model['classifier'].feature_importances_
    
    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df

feature_importance_df = get_feature_importance(model)
if feature_importance_df is not None:
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Features for Predicting Email Clicks')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nFeature importance visualization saved as 'feature_importance.png'")

# Answer Question 3: Expected improvement in click-through rate
print("\n----- Question 3: Expected CTR Improvement -----")

# Make predictions on test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate baseline click rate
baseline_click_rate = y_test.mean() * 100
print(f"Baseline click rate (random targeting): {baseline_click_rate:.2f}%")

# Calculate potential improvements at different targeting thresholds
percentiles = [0.1, 0.2, 0.3, 0.5, 0.7]
improvements = []

scored_users = pd.DataFrame({
    'actual': y_test,
    'pred_proba': y_pred_proba
}).sort_values('pred_proba', ascending=False)

print("\nPotential improvements by targeting top users:")
print(f"{'Targeting Top':20} {'Predicted CTR':15} {'Improvement':15}")
print("-" * 50)

for p in percentiles:
    top_p_count = int(len(scored_users) * p)
    top_p = scored_users.head(top_p_count)
    targeted_click_rate = top_p['actual'].mean() * 100
    improvement = (targeted_click_rate / baseline_click_rate - 1) * 100
    
    print(f"{f'{p*100:.0f}% of users':20} {f'{targeted_click_rate:.2f}%':15} {f'+{improvement:.1f}%':15}")
    
    improvements.append({
        'percentile': p * 100,
        'targeted_rate': targeted_click_rate,
        'improvement': improvement
    })

# Visualize improvements
plt.figure(figsize=(10, 6))
improvement_df = pd.DataFrame(improvements)
sns.lineplot(x='percentile', y='improvement', data=improvement_df, marker='o')
plt.title('Expected CTR Improvement vs. Targeting Percentage')
plt.xlabel('Top Percentage of Users Targeted')
plt.ylabel('Improvement %')
plt.grid(True)
plt.tight_layout()
plt.savefig('ctr_improvement.png')
print("\nCTR improvement visualization saved as 'ctr_improvement.png'")

# A/B Testing Recommendation
print("\nHow to test the model:")
print("""
1. Design an A/B test:
   - Group A (Control): Random sample of users (current approach)
   - Group B (Test): Users selected by the model (e.g., top 20% by predicted probability)
   
2. Implementation:
   - Ensure groups are of equal size and statistically comparable
   - Send identical emails to both groups at the same time
   - Measure CTR in both groups
   
3. Statistical validation:
   - Use chi-square test to verify if the difference is statistically significant
   - Calculate p-value and confidence intervals
   
4. Sample size estimation:
   - For detecting a 50% improvement with 80% power and 95% confidence:
     If baseline CTR is around 2%, would need approximately 2,000 users per group
""")

# Answer Question 4: Interesting patterns for different segments
print("\n----- Question 4: Interesting Patterns by User Segment -----")

# Create a function to analyze patterns for a specific factor
def analyze_segment(factor):
    segment_data = email_table.groupby(factor).agg(
        total=('email_id', 'count'),
        opened=('opened', 'sum'),
        clicked=('clicked', 'sum')
    ).reset_index()
    
    segment_data['open_rate'] = segment_data['opened'] / segment_data['total'] * 100
    segment_data['click_rate'] = segment_data['clicked'] / segment_data['total'] * 100
    
    return segment_data

# Analyze patterns by different segments
print("\n1. Performance by Email Text Length:")
text_performance = analyze_segment('email_text')
print(text_performance[['email_text', 'total', 'open_rate', 'click_rate']])

print("\n2. Performance by Email Personalization:")
version_performance = analyze_segment('email_version')
print(version_performance[['email_version', 'total', 'open_rate', 'click_rate']])

print("\n3. Performance by Hour of Day (Top 5):")
hour_performance = analyze_segment('hour')
print(hour_performance.sort_values('click_rate', ascending=False).head()[['hour', 'total', 'click_rate']])

print("\n4. Performance by Weekday:")
weekday_performance = analyze_segment('weekday')
print(weekday_performance[['weekday', 'total', 'click_rate']])

# Create purchase buckets for better analysis
email_table['purchase_bucket'] = pd.cut(
    email_table['user_past_purchases'], 
    bins=[-1, 0, 2, 5, 10, float('inf')],
    labels=['0', '1-2', '3-5', '6-10', '10+']
)

print("\n5. Performance by Purchase History:")
purchase_performance = analyze_segment('purchase_bucket')
print(purchase_performance[['purchase_bucket', 'total', 'click_rate']])

# Find the best combination of factors
print("\n6. Best Combinations (Top 5):")
combination_df = email_table.groupby(['email_text', 'email_version', 'purchase_bucket']).agg(
    total=('email_id', 'count'),
    clicked=('clicked', 'sum')
).reset_index()

combination_df['click_rate'] = combination_df['clicked'] / combination_df['total'] * 100
# Only consider combinations with at least 20 emails
significant_combos = combination_df[combination_df['total'] >= 20]
print(significant_combos.sort_values('click_rate', ascending=False).head()[
    ['email_text', 'email_version', 'purchase_bucket', 'total', 'click_rate']
])

# Create visualizations for the most interesting patterns
plt.figure(figsize=(16, 12))

# Plot 1: Click rates by hour
plt.subplot(2, 2, 1)
sns.barplot(x='hour', y='click_rate', data=hour_performance.sort_values('hour'))
plt.title('Click Rate by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Click Rate (%)')
plt.xticks(rotation=90)

# Plot 2: Click rates by weekday
plt.subplot(2, 2, 2)
sns.barplot(x='weekday', y='click_rate', data=weekday_performance)
plt.title('Click Rate by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Click Rate (%)')

# Plot 3: Click rates by email type and version
plt.subplot(2, 2, 3)
combined_factors = email_table.groupby(['email_text', 'email_version']).agg(
    click_rate=('clicked', lambda x: sum(x) / len(x) * 100)
).reset_index()
sns.barplot(x='email_text', y='click_rate', hue='email_version', data=combined_factors)
plt.title('Click Rate by Email Type and Version')
plt.xlabel('Email Text Length')
plt.ylabel('Click Rate (%)')

# Plot 4: Click rates by purchase history
plt.subplot(2, 2, 4)
sns.barplot(x='purchase_bucket', y='click_rate', data=purchase_performance)
plt.title('Click Rate by Purchase History')
plt.xlabel('Past Purchase Count')
plt.ylabel('Click Rate (%)')

plt.tight_layout()
plt.savefig('segment_patterns.png')
print("\nSegment patterns visualization saved as 'segment_patterns.png'")

# Check for interaction between time and user type
print("\n7. Timing and User Purchase History Interaction:")
time_purchase_df = email_table.groupby(['hour', 'purchase_bucket']).agg(
    total=('email_id', 'count'),
    clicked=('clicked', 'sum')
).reset_index()
time_purchase_df['click_rate'] = time_purchase_df['clicked'] / time_purchase_df['total'] * 100
time_purchase_df = time_purchase_df[time_purchase_df['total'] >= 15]  # Filter for significance

print("Best timing for high-value customers (6+ past purchases):")
high_value = time_purchase_df[time_purchase_df['purchase_bucket'].isin(['6-10', '10+'])]
print(high_value.sort_values('click_rate', ascending=False).head()[['hour', 'purchase_bucket', 'total', 'click_rate']])

print("\nSummary of key findings:")
print("1. Email content and personalization impact: ", end="")
if len(combined_factors) > 0:
    best_format = combined_factors.loc[combined_factors['click_rate'].idxmax()]
    print(f"{best_format['email_text']} text with {best_format['email_version']} format performs best")

print("2. Timing matters: ", end="")
if len(hour_performance) > 0:
    best_hour = hour_performance.loc[hour_performance['click_rate'].idxmax()]
    print(f"Hour {best_hour['hour']} shows highest engagement at {best_hour['click_rate']:.2f}% CTR")

print("3. User purchase history correlation: ", end="")
if len(purchase_performance) > 0:
    best_purchase = purchase_performance.loc[purchase_performance['click_rate'].idxmax()]
    print(f"Users with {best_purchase['purchase_bucket']} past purchases have {best_purchase['click_rate']:.2f}% CTR")

print("\nRecommended targeting strategy based on analysis:")
print("- Focus on users with specific purchase patterns")
print("- Send emails at optimal times based on user segments")
print("- Use the right combination of content length and personalization")
print(f"- Expected improvement: Targeting optimally could increase CTR by up to {improvement_df['improvement'].max():.1f}%")