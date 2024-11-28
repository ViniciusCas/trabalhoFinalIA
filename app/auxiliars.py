import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns 

class EDA:
    
    def __init__(self, df, normal_df):
        self.df = df
        self.normal_df = normal_df
    
    def plot_correlation(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.normal_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlação entre as variáveis")
        return fig, ax

    # def 
    # 10010