from scipy import stats
import numpy as np
import pandas as pd

""" 
Discrete Probability Calculation - Binomial Theorem.
"""
def fact(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def combination(n, r):
    comb = int(fact(n) / (fact(n - r) * fact(r)))
    return comb

def binom_pmf(n, p, r):
    res = combination(n, r) * (p ** r) * ((1 - p) ** (n - r))
    return res
## Sample Problem   - Binomial Theorem
n=7
p=1/6
r=2
probability = binom_pmf(n, p, r)
print(round(probability, 2))

"""
Continuous Probability Calculation - Normal Distribution
"""

def normald(lower_bound=-3, upper_bound=3, mean=0, std_dev=1):
    
   
    # Calculate the Z-scores for the lower and upper bounds
    z_lower = (lower_bound - mean) / std_dev
    z_upper = (upper_bound - mean) / std_dev

    # Calculate the cumulative probability within the range using Z-scores
    probability = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)

    print(f"Probability of falling within [{lower_bound}, {upper_bound}]: {probability:.4f}")

# Sample Example 
lower_bound=0
upper_bound=70
mean=80
std_dev=20
normald(lower_bound, upper_bound, mean, std_dev)


"""
Central Limit Theorem or Margin of Error/ Mean Estimation
"""

def normal_sample(std_dev, confidence_level=0.95, sample_size=100, sample_mean=None, digits=2):
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    SE = std_dev / np.sqrt(sample_size)
    MOE = round(z_score * SE, digits)
    print(f"Margin of Error for Confidence Level {confidence_level*100}% : {MOE}")
    if sample_mean:
        moe_interval = (round(sample_mean-MOE, digits),round(sample_mean+MOE, digits))
        print(f"Mean Interval for Confidence Level {confidence_level*100}% : {moe_interval}")
        

# Sample Example - For 90% confidence and mean = 8 , find mean intervals
std_dev=2
confidence_level = 0.90
sample_size=100
sample_mean=8
normal_sample(std_dev, confidence_level, sample_size, sample_mean)

# o/p: 
# Margin of Error for Confidence Level 90.0% : 0.33
# Mean Interval for Confidence Level 90.0% : (7.67, 8.33)


"""
Hypothesis Testing - Critical Value Method.      
"""

def critical_value_method(sample_mean=None, sample_stddev=None, sample_size=None, data=None, null_mean=None, alpha=0.05, test_type='two-tailed', _round=2):
    if data is not None:
        sample_mean = round(np.mean(data), _round)
        sample_stddev = round(np.std(data, ddof=1), _round)
        sample_size = len(data)

    print(f"sample_mean={sample_mean}, sample_stddev={sample_stddev}, sample_size={sample_size}")

    if test_type == 'two-tailed':
        z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        z_critical_upper = z_alpha_2
        z_critical_lower = -z_alpha_2

        value_at_z_critical_upper = null_mean + z_critical_upper * (sample_stddev / np.sqrt(sample_size))
        value_at_z_critical_lower = null_mean + z_critical_lower * (sample_stddev / np.sqrt(sample_size))

        print(f"z_critical_upper={z_critical_upper:.4f}, z_critical_lower={z_critical_lower:.4f}, "
              f"value_at_z_critical_upper={value_at_z_critical_upper:.4f}, value_at_z_critical_lower={value_at_z_critical_lower:.4f}")

        if sample_mean > value_at_z_critical_upper or sample_mean < value_at_z_critical_lower:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    elif test_type == 'left':
        z_alpha = stats.norm.ppf(alpha)
        z_critical_lower = -z_alpha

        value_at_z_critical_lower = null_mean + z_critical_lower * (sample_stddev / np.sqrt(sample_size))

        print(f"z_critical_lower={z_critical_lower:.4f}, "
              f"value_at_z_critical_lower={value_at_z_critical_lower:.4f}")

        if sample_mean < value_at_z_critical_lower:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    elif test_type == 'right':
        z_alpha = stats.norm.ppf(1 - alpha)
        z_critical_upper = z_alpha

        value_at_z_critical_upper = null_mean + z_critical_upper * (sample_stddev / np.sqrt(sample_size))

        print(f"z_critical_upper={z_critical_upper:.4f}, "
              f"value_at_z_critical_upper={value_at_z_critical_upper:.4f}")

        if sample_mean > value_at_z_critical_upper:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    else:
        raise ValueError("Invalid test_type. Use 'two-tailed', 'left', or 'right'.")

# Example:
null_mean = 36
alpha = 0.03
sample_mean = 34.5
sample_size = 49
sample_std = 4
test_type = 'two-tailed'
print(critical_value_method(sample_mean=sample_mean,sample_stddev=sample_std,sample_size=sample_size, null_mean=null_mean, alpha=alpha,test_type=test_type))

# Output:
# sample_mean=34.5, sample_stddev=4, sample_size=49
# z_critical_upper=2.1701, z_critical_lower=-2.1701, value_at_z_critical_upper=37.2401, value_at_z_critical_lower=34.7599
# Reject the null hypothesis


"""
Hypothesis Testing - P-value Test Method.      
"""


def p_value_method(sample_mean=None, sample_stddev=None, sample_size=None, data=None, null_mean=None, alpha=0.05, test_type='two-tailed'):
 
    if data is not None:
        sample_mean = round(np.mean(data), 2)
        sample_stddev = round(np.std(data, ddof=1), 2)
        sample_size = len(data)
    
    print(f"sample_mean={sample_mean}, sample_stddev={sample_stddev},sample_size={sample_size}")

    if sample_mean is None or sample_stddev is None or sample_size is None:
        raise ValueError("Either provide 'data' or 'sample_mean', 'sample_stddev', 'sample_size', 'null_mean', and 'alpha'.")

    if test_type == 'two-tailed':
        t_statistic = (sample_mean - null_mean) / (sample_stddev / np.sqrt(sample_size))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=sample_size-1))
        print(f"p_value={p_value:.4f} and Z-Score={t_statistic:.4f}")
        if p_value < alpha:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    elif test_type == 'left':
        t_statistic = (sample_mean - null_mean) / (sample_stddev / np.sqrt(sample_size))
        p_value = stats.t.cdf(t_statistic, df=sample_size-1)
        print(f"p_value={p_value:.4f} and Z-Score={t_statistic:.4f}")
        if p_value < alpha:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    elif test_type == 'right':
        t_statistic = (sample_mean - null_mean) / (sample_stddev / np.sqrt(sample_size))
        p_value = 1 - stats.t.cdf(t_statistic, df=sample_size -1)
        print(f"p_value={p_value:.4f} and Z-Score={t_statistic:.4f}")
        if p_value < alpha:
            return "Reject the null hypothesis"
        else:
            return "Fail to reject the null hypothesis"
    else:
        raise ValueError("Invalid test_type. Use 'two-tailed', 'left', or 'right'.")
    
# Example:  Null Hpythosis.
null_mean = 0.6
alpha = 0.05
df = np.arry([1,0,1,1,1,1,1,1,1,1,1,0,1]*10)
print(p_value_method(data=df, null_mean=null_mean, alpha=alpha, test_type='left'))

# Output
# sample_mean=0.52, sample_stddev=0.5,sample_size=121
# p_value=0.0405 and Z-Score=-1.7600
# Reject the null hypothesis
