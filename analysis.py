import os
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np


def load_data(file_name: str) -> pd.DataFrame:
    """
    Load JSON data into a pandas dataframe.

    :param file_name: Name of the file containing the data.
    :return: A pandas dataframe with the data.
    """
    return pd.read_json(file_name)

def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a dataframe with nested dictionaries.

    :param df: A pandas dataframe with nested structures.
    :return: A flattened dataframe.
    """
    df_flat = pd.json_normalize(df.to_dict(orient='records'))
    return df_flat

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the data.

    :param df: A pandas dataframe with the data.
    :return: A dataframe containing the means, medians, and standard deviations.
    """
    # Descriptive stats
    means = df.mean()
    medians = df.median()
    std_devs = df.std()

    # Combine into one dataframe
    desc_stats = pd.concat([means, medians, std_devs], axis=1)
    desc_stats.columns = ['Mean', 'Median', 'Standard Deviation']

    return desc_stats

def inferential_statistics(df: pd.DataFrame) -> dict:
    """
    Perform inferential statistics (paired t-tests and effect sizes) on the data.

    :param df: A pandas dataframe with the data.
    :return: A dictionary with t-test results and effect sizes.
    """
    # Assuming columns have a structure like 'generic_content_C1_concept_relevance' and 'personalized_content_C1_concept_relevance'
    results = {}
    for column in df.columns:
        if 'generic_content' in column:
            personalized_col = column.replace('generic_content', 'personalized_content')
            t_stat, p_value = stats.ttest_rel(df[column], df[personalized_col])
            
            # Calculating effect size (Cohen's d)
            diff = df[column].mean() - df[personalized_col].mean()
            pooled_std = (df[column].std() + df[personalized_col].std()) / 2
            effect_size = diff / pooled_std
            
            results[column] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size
            }

    return results

def perform_anova(df: pd.DataFrame, demographic_var: str, content_col: str) -> float:
    """
    Perform an ANOVA test for a demographic variable against a content column.

    :param df: Flattened and numeric dataframe.
    :param demographic_var: Demographic variable to test against (e.g., "demographics.student_major").
    :param content_col: The content column (e.g., "generic_content.P3.personalize_effectively").
    :return: p-value from the ANOVA test.
    """
    formula = 'Q("' + content_col + '") ~ C(Q("' + demographic_var + '"))'
    model = ols(formula, df).fit()
    aov_table = anova_lm(model, typ=2)
    return aov_table["PR(>F)"][0]

def run_anova_tests(df: pd.DataFrame) -> dict:
    """
    Run ANOVA tests for each content column based on demographic variables.

    :param df: The dataframe containing the data.
    :return: A dictionary containing p-values for each ANOVA test.
    """
    demographic_vars = ["demographics.student_major", "demographics.prior_expertise", "demographics.perception_of_CS_value"]
    content_columns = [col for col in df.columns if "generic_content" in col or "personalized_content" in col]

    results = {}
    for demo_var in demographic_vars:
        results[demo_var] = {}
        for content_col in content_columns:
            p_value = perform_anova(df, demo_var, content_col)
            results[demo_var][content_col] = p_value

    return results

def create_visualizations(df: pd.DataFrame):
    """
    Create enhanced visualizations for the data and save them.

    :param df: A pandas dataframe with the data.
    """
    
    # Ensure the directory exists
    if not os.path.exists("Visualizations"):
        os.mkdir("Visualizations")

    content_columns = [col for col in df.columns if "generic_content" in col or "personalized_content" in col]
    
    for column in content_columns:
        generic_col = column
        personalized_col = column.replace('generic_content', 'personalized_content')
        
        # Bar Plots with Error Bars
        plt.figure(figsize=(10,6))
        means = [df[generic_col].mean(), df[personalized_col].mean()]
        stds = [df[generic_col].std(), df[personalized_col].std()]
        sns.barplot(x=['Generic', 'Personalized'], y=means, yerr=stds)
        plt.title(f"{column}_barplot")
        plt.ylabel('Mean Value')
        plt.savefig(f"Visualizations/{column}_barplot.png")
        plt.close()

        # Box Plots by Major
        df_melted = df.melt(value_vars=[generic_col, personalized_col], 
                            id_vars="demographics.student_major", 
                            var_name="Content Type", value_name="Value")
        plt.figure(figsize=(12,8))
        sns.boxplot(x="demographics.student_major", y="Value", hue="Content Type", data=df_melted)
        plt.title(f"{column}_boxplot_by_major")
        plt.savefig(f"Visualizations/{column}_boxplot_by_major.png")
        plt.close()

    
        # Violin Plots by Major
        df_melted = df.melt(value_vars=[generic_col, personalized_col], 
                            id_vars="demographics.student_major", 
                            var_name="Content Type", value_name="Value")
        
        plt.figure(figsize=(12,8))
        
        if df_melted["Content Type"].nunique() == 2:  # Check if there are exactly two levels
            sns.violinplot(x="demographics.student_major", y="Value", hue="Content Type", data=df_melted, split=True)
        else:
            sns.violinplot(x="demographics.student_major", y="Value", hue="Content Type", data=df_melted)
            
        plt.title(f"{column}_violinplot_by_major")
        plt.savefig(f"Visualizations/{column}_violinplot_by_major.png")
        plt.close()

        # Histogram or Density Plots
        plt.figure(figsize=(10,6))
        sns.kdeplot(df[generic_col], label="Generic", shade=True)
        sns.kdeplot(df[personalized_col], label="Personalized", shade=True)
        plt.title(f"{column}_densityplot")
        plt.ylabel('Density')
        plt.xlabel('Value')
        plt.savefig(f"Visualizations/{column}_densityplot.png")
        plt.close()

    # Effect Size Visualization (Cohen's d)
    effect_sizes = {}
    for column in content_columns:
        if "generic_content" in column:
            diff = df[column].mean() - df[column.replace('generic_content', 'personalized_content')].mean()
            pooled_std = np.sqrt((df[column].var() + df[column.replace('generic_content', 'personalized_content')].var()) / 2)
            effect_size = diff / pooled_std
            effect_sizes[column] = effect_size
            
    plt.figure(figsize=(12,6))
    names = list(effect_sizes.keys())
    values = list(effect_sizes.values())
    sns.barplot(x=names, y=values)
    plt.title("Effect Sizes (Cohen's d) for each Content Column")
    plt.xticks(rotation=90)
    plt.ylabel("Effect Size (Cohen's d)")
    plt.savefig("Visualizations/effect_sizes.png")
    plt.close()

    # Lastly, the heatmap for ANOVA p-values would require the result from the ANOVA tests,
    # so it might be better placed in the main function or where the ANOVA tests are run.

print("Visualizations have been saved to the 'Visualizations' directory.")


def main(file_name: str):
    """
    Main function to run the analysis.

    :param file_name: Name of the file containing the data.
    """
    # Load and flatten data
    df = load_data(file_name)
    df_flat = flatten_dataframe(df)
    df_vis = flatten_dataframe(df)
    df_flat["demographics.student_major"] = pd.Categorical(df_flat["demographics.student_major"]).codes
    category_mapping = dict(enumerate(pd.Categorical(df_flat["demographics.student_major"]).categories))
    print(category_mapping)
    

    # Exclude non-numeric columns
    numeric_columns = df_flat.select_dtypes(include=['number']).columns
    df_numeric = df_flat[numeric_columns]
    print(df_numeric.columns)


    # Descriptive Statistics
    desc_stats = descriptive_statistics(df_numeric)  # Use the numeric-only dataframe
    print(desc_stats)

    # Inferential Statistics
    infer_stats = inferential_statistics(df_numeric)
    for column, stats in infer_stats.items():
        print(f"\nColumn: {column}")
        print(f"T-statistic: {stats['t_statistic']}\nP-value: {stats['p_value']}\nEffect Size: {stats['effect_size']}")

    # ANOVA Tests
    anova_results = run_anova_tests(df_numeric)
    for demo_var, content_results in anova_results.items():
        print(f"\nDemographic Variable: {demo_var}")
        for content_col, p_value in content_results.items():
            print(f"{content_col}: p-value = {p_value}")

    # Visualizations
    create_visualizations(df_vis)

    # Save results to a file
    with open('synthetic_results.txt', 'w') as f:
        f.write(str(desc_stats))
        f.write('\n\n')
        for column, stats in infer_stats.items():
            f.write(f"Column: {column}\nT-statistic: {stats['t_statistic']}\nP-value: {stats['p_value']}\nEffect Size: {stats['effect_size']}\n\n")
        f.write('\n\nANOVA Results\n')
        for demo_var, content_results in anova_results.items():
            f.write(f"\nDemographic Variable: {demo_var}\n")
            for content_col, p_value in content_results.items():
                f.write(f"{content_col}: p-value = {p_value}\n")

    print("\nResults have been saved to 'synthetic_results.txt'.")

# Run the main function
if __name__ == '__main__':
    main('synthetic_data.json')  # Replace 'your_file_name.json' with your actual file name

