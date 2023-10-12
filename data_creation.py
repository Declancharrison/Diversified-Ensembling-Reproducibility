import folktables
import numpy as np
import sklearn as sk

ACSIncome = folktables.BasicProblem(
    features=[
        "ST",
        "AGEP",
        "CIT",
        "COW",
        "DDRS",
        "DEAR",
        "DEYE",
        "DOUT",
        "DRAT",
        "DREM",
        "ENG",
        "FER",
        "JWTRNS",
        "LANX",
        "MAR",
        "MIL",
        "SCHL",
        "SEX",
        'WKHP',
        "OCCP",
        "RAC1P"
    ],
    target='PINCP',
    # target_transform=lambda x: x > 50000,    
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

data_source = folktables.ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["VA","TX", "WV", "KY", "FL", "OK", "TN", "AK", "SC", "AL", "NC", "LA", "MS", "MD", "GA", "DE"], download=True)
data_np, labels, _ = ACSIncome.df_to_numpy(acs_data)
indices = labels < 100000
data_np = data_np[indices]
labels = labels[indices]
data = pd.DataFrame(data_np, columns = ["ST", "AGEP", "CIT", "COW", "DDRS", "DEAR", "DEYE", "DOUT", "DRAT", "DREM", "ENG", "FER", "JWTRNS", "LANX", "MAR", "MIL", "SCHL", "SEX", 'WKHP', "OCCP", 'RAC1P'])

x_train, x_val_test, y_train, y_val_test = sk.metrics.train_test_split(data, labels, test_size = .3, random_state = 23)
x_val, x_test, y_val, y_test = sk.metrics.train_test_split(x_val_test, y_val_test, test_size = .5, random_state = 23)

x_train.reset_index(drop=True, inplace = True)
x_val.reset_index(drop=True, inplace = True)
x_test.reset_index(drop=True, inplace = True)

x_train.to_csv('data/training_data.csv', index=False)
y_train.tofile('data/training_labels.csv', sep =",")
x_val.to_csv('data/validation_data.csv', index=False)
y_val.tofile('data/validation_labels.csv', sep =",")
x_test.to_csv('data/test_data.csv', index=False)
y_test.tofile('data/test_labels.csv', sep =",")