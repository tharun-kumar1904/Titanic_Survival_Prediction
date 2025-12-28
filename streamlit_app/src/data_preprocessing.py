import pandas as pd
import numpy as np

def preprocess(df, is_train=True, feature_columns=None):
    df = df.copy()
    passenger_ids = df['PassengerId'] if 'PassengerId' in df.columns else None

    # -------------------------
    # Missing values
    # -------------------------
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # -------------------------
    # Feature engineering
    # -------------------------
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

    # -------------------------
    # Drop unused columns
    # -------------------------
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

    # -------------------------
    # Encoding
    # -------------------------
    df = pd.get_dummies(
        df,
        columns=['Sex', 'Embarked', 'Title', 'Pclass'],
        drop_first=True
    )

    # -------------------------
    # Feature consistency
    # -------------------------
    if is_train:
        feature_columns = df.columns.tolist()
    else:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    if passenger_ids is not None and 'PassengerId' not in df.columns:
        df.insert(0, 'PassengerId', passenger_ids)

    return df, feature_columns
