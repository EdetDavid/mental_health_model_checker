import pandas as pd
import numpy as np

# Create directory to store the student mental health dataset
# This is synthetic data based on common mental health survey attributes

# Create a synthetic dataset with common mental health indicators
np.random.seed(42)
n_samples = 500

# Generate data
data = {
    # Demographics
    'age': np.random.normal(20, 2, n_samples).clip(17, 30).astype(int),
    'gender': np.random.choice(['Female', 'Male', 'Non-binary'], n_samples, p=[0.55, 0.42, 0.03]),
    'international': np.random.choice(['Yes', 'No'], n_samples, p=[0.18, 0.82]),
    'year_of_study': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'marital_status': np.random.choice(['Single', 'In a relationship', 'Married', 'Divorced'], 
                                     n_samples, p=[0.7, 0.25, 0.04, 0.01]),
    
    # Academic factors
    'gpa': np.random.normal(3.2, 0.6, n_samples).clip(0, 4.0).round(2),
    'major': np.random.choice(['Engineering', 'Arts', 'Science', 'Business', 'Medicine', 
                             'Education', 'Social Sciences', 'Computer Science'], n_samples),
    'course_load': np.random.choice([3, 4, 5, 6, 7], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
    
    # Health indicators
    'sleep_hours': np.random.normal(6.7, 1.5, n_samples).clip(3, 10).round(1),
    'exercise_weekly': np.random.choice(['None', '1-2 days', '3-4 days', '5+ days'], 
                                      n_samples, p=[0.25, 0.40, 0.25, 0.1]),
    
    # Mental health screening
    'depression_score': np.random.choice(range(28), n_samples),  # PHQ-9 scale (0-27)
    'anxiety_score': np.random.choice(range(22), n_samples),  # GAD-7 scale (0-21)
    'stress_level': np.random.choice(['Low', 'Moderate', 'High', 'Severe'], 
                                   n_samples, p=[0.2, 0.4, 0.3, 0.1]),
    
    # Support systems
    'counseling_before': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    'family_support': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], 
                                    n_samples, p=[0.1, 0.2, 0.4, 0.3]),
    'friend_support': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], 
                                   n_samples, p=[0.05, 0.15, 0.5, 0.3]),
    
    # External stressors
    'financial_stress': np.random.choice(['None', 'Low', 'Moderate', 'High', 'Severe'], 
                                      n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'living_condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], 
                                      n_samples, p=[0.1, 0.3, 0.4, 0.2]),
    
    # Device and social media usage
    'screen_time_daily': np.random.choice(['1-2 hours', '3-5 hours', '6-8 hours', '9+ hours'], 
                                       n_samples, p=[0.1, 0.3, 0.4, 0.2]),
    'social_media_daily': np.random.choice(['Less than 1 hour', '1-2 hours', '3-4 hours', '5+ hours'], 
                                        n_samples, p=[0.15, 0.35, 0.3, 0.2])
}

# Create dataframe
df = pd.DataFrame(data)

# Make depression and anxiety scores follow a more realistic distribution
df['depression_score'] = np.random.gamma(1.5, 2.0, n_samples).clip(0, 27).astype(int)
df['anxiety_score'] = np.random.gamma(1.2, 2.0, n_samples).clip(0, 21).astype(int)

# Convert depression and anxiety scores to 0-100 scale
df['depression_normalized'] = 100 - (df['depression_score'] * 100 / 27)  # Higher is better
df['anxiety_normalized'] = 100 - (df['anxiety_score'] * 100 / 21)       # Higher is better

# Convert stress level to numeric
stress_map = {'Low': 75, 'Moderate': 50, 'High': 25, 'Severe': 0}       # Higher is better
df['stress_numeric'] = df['stress_level'].map(stress_map)

# Convert support systems to numeric
support_map = {'Poor': 0, 'Fair': 33, 'Good': 67, 'Excellent': 100}
df['family_support_numeric'] = df['family_support'].map(support_map)
df['friend_support_numeric'] = df['friend_support'].map(support_map)

# Create mental health index (weighted average)
df['mental_health_index'] = (
    df['depression_normalized'] * 0.35 +
    df['anxiety_normalized'] * 0.35 +
    df['stress_numeric'] * 0.15 +
    df['family_support_numeric'] * 0.075 +
    df['friend_support_numeric'] * 0.075
).round(1)

# Create mental health category
df['mental_health_category'] = pd.cut(
    df['mental_health_index'],
    bins=[0, 40, 60, 100],
    labels=['Poor', 'Average', 'Good']
)

# Drop intermediate columns
df = df.drop(['depression_normalized', 'anxiety_normalized', 'stress_numeric', 
            'family_support_numeric', 'friend_support_numeric'], axis=1)

# Save the dataset
df.to_csv('/home/dvooskid/Desktop/mental_health_model/data/student_mental_health.csv', index=False)
print("Dataset created and saved to: /home/dvooskid/Desktop/mental_health_model/data/student_mental_health.csv")
print(f"Sample size: {len(df)} records")
print("\nData preview:")
print(df.head())
print("\nData summary:")
print(df.describe())
