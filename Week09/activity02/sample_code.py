from ucimlrepo import fetch_ucirepo

# fetch dataset
user_knowledge_modeling = fetch_ucirepo(id=257)

# data (as pandas dataframes)
X = user_knowledge_modeling.data.features
y = user_knowledge_modeling.data.targets

# metadata
print(user_knowledge_modeling.metadata)

# variable information
print(user_knowledge_modeling.variables)
