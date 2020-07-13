from sklearn import tree
import pandas as pd
from joblib import dump, load

train_csv = pd.read_csv("../../Data/train.csv")
test_csv = pd.read_csv("../../Data/test.csv")
train_data = []
train_y = []
maxn = 33126
testn = 500

patients = {}
images = {}

# male = 1, female = 0; torso = 0, lower extremity = 1, upper extremity = 2, head/neck = 3, palms/soles = 4, oral/genital = 5

# Train
for i in range(maxn-testn):
    try:
        sex = 1 if train_csv['sex'][i] == 'male' else 0
        anatom = train_csv['anatom_site_general_challenge'][i]
        site = 0
        if(anatom == 'torso'):
            site = 0
        elif(anatom == 'lower extremity'):
            site = 1
        elif(anatom == 'upper extremity'):
            site = 2
        elif(anatom == 'head/neck'):
            site = 3
        elif(anatom == 'palms/soles'):
            site = 4
        elif(anatom == 'oral/genital'):
            site = 5
        else:
            site = 0

        age_approx = train_csv['age_approx'][i]
        age = 45

        try:
            age = float(age_approx) if (float(age_approx) < 100) else 45
        except:
            age = 45

        patient = train_csv['patient_id'][i]
        if(patient not in patients):
            patients[patient] = len(patients)

        image = train_csv['image_name'][i]
        if(image not in images):
            images[image] = len(images)

        train_data.append([images[image], patients[patient], sex, age, site])

        y = 0
        try:
            y = float(train_csv['target'][i])
        except:
            y = 0
        train_y.append(y)
    except:
        print(i, "Malfunctioning data")
        break

# Fit
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_y)
# Export tree
dump(clf, 'ID3_Melanoma.d')
print("ok")
