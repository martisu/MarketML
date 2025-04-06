import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
file_path = 'marketing_campaign.csv'
data = pd.read_csv(file_path, delimiter=';')