import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

class ABTesting:
    def __init__(self, data):
        self.data = data

    def test_risk_across_provinces(self):
        print("\nTesting risk differences across provinces...")
        
        contingency_table = pd.crosstab(self.data["Province"], self.data["TotalClaims"] > 0)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        if p_value < 0.05:
            print("Reject the null hypothesis. There are significant risk differences across provinces.")
        else:
            print("Fail to reject the null hypothesis. There are no significant risk differences across provinces.")

    def test_risk_between_zipcodes(self):
        print("\nTesting risk differences between zip codes...")
        
        contingency_table = pd.crosstab(self.data["PostalCode"], self.data["TotalClaims"] > 0)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        if p_value < 0.05:
            print("Reject the null hypothesis. There are significant risk differences between zip codes.")
        else:
            print("Fail to reject the null hypothesis. There are no significant risk differences between zip codes.")

    def test_margin_difference_zipcodes(self):
        """
        Test the null hypothesis: There are no significant margin differences between zip codes.
        """
        print("\nTesting margin differences between zip codes...")
        
        profit_by_zipcode = (
            self.data.groupby("PostalCode")["TotalPremium"].sum() -
            self.data.groupby("PostalCode")["TotalClaims"].sum()
        )

        zipcodes_even = profit_by_zipcode.loc[profit_by_zipcode.index % 2 == 0]
        zipcodes_odd = profit_by_zipcode.loc[profit_by_zipcode.index % 2 != 0]

        t_stat, p_value = ttest_ind(zipcodes_even, zipcodes_odd, nan_policy='omit')

        if p_value < 0.05:
            print("Reject the null hypothesis. There are significant margin differences between zip codes.")
        else:
            print("Fail to reject the null hypothesis. There are no significant margin differences between zip codes.")

    def test_risk_by_gender(self):
        print("\nTesting risk differences between genders...")
        
        contingency_table = pd.crosstab(self.data["Gender"], self.data["TotalClaims"] > 0)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        if p_value < 0.05:
            print("Reject the null hypothesis. There are significant risk differences between genders.")
        else:
            print("Fail to reject the null hypothesis. There are no significant risk differences between genders.")

    def perform_all_tests(self):
        self.test_risk_across_provinces()
        self.test_risk_between_zipcodes()
        self.test_margin_difference_zipcodes()
        self.test_risk_by_gender()