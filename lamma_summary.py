import ollama

def give_summary(num_cols,cat_cols,nulls_summary,duplicates_count,short_cat_cols,short_num_cols,short_nulls):
    prompt = f"""
                Analyze this dataset for clustering Overview.

                Numerical Columns (partial): {short_num_cols}
                Categorical Columns (partial): {short_cat_cols}
                Null Values (partial): {short_nulls}
                Duplicates: {duplicates_count}

                Provide an summary in the format of 4 Headings 1.Data Insights 2.Data Quality 3.Feature Engineering and Transformation Suggession 4.Clustering Model Suggession
                In this headings provide complete data bagged infromation in the form of multiline paragraph and bullet points not more than 6
                It should be completely data bagged means related to data not anything outside of it.
                """
    response = ollama.chat(
                        model="llama3.2:latest",
                        messages=[{"role": "user", "content": prompt}]
                    )

    summary_text = response["message"]["content"].replace("•", "-")
    
    return summary_text
    