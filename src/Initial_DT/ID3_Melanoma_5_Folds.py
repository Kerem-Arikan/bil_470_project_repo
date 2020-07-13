from sklearn import tree
import pandas as pd
import threading
import time


# male = 1, female = 0; torso = 0, lower extremity = 1, upper extremity = 2, head/neck = 3, palms/soles = 4, oral/genital = 5

success = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
success_G = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
predict_cancer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def ID3(test_area, train_csv, maxn):
    patients = {}
    images = {}
    train_data = []
    train_y = []
    # Train
    for i in [k for k in range(maxn) if (not (k >= (maxn/20)*test_area and k < (maxn/20)*(test_area+1)))]:
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

            train_data.append(
                [images[image], patients[patient], sex, age, site])

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

    # Test
    for i in [k for k in range(maxn) if (k >= (maxn/20)*test_area and k < (maxn/20)*(test_area+1))]:
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

            y = 0
            try:
                y = float(train_csv['target'][i])
            except:
                y = 0

            predict = clf.predict(
                [[images[image], patients[patient], sex, age, site]])
            if(y == 1):
                total[test_area] += 1
                if(predict == 1):
                    success[test_area] += 1
            if(y == predict):
                success_G[test_area] += 1

            if(predict == 1):
                predict_cancer[test_area] += 1
        except:
            print(i, "Malfunctioning data")
            break


train_csv = pd.read_csv("../../Data/train.csv")
maxn = 33126
maxn = 30000

# Five folds testing with threads
threads = []
for test_area in range(20):
    t = threading.Thread(target=ID3, args=(test_area, train_csv, maxn))
    threads.append(t)

for t in threads:
    t.start()


success_sum = 0  # Amount of cancer hits
total_sum = 0  # Amount of cancer in data
success_G_sum = 0  # Amount of all hits in data
predict_cancer_sum = 0
for i in range(20):
    threads[i].join()
    success_sum += success[i]
    total_sum += total[i]
    success_G_sum += success_G[i]
    predict_cancer_sum += predict_cancer[i]

print("Cancer Hit===", "success_rate:", 0 if (total_sum == 0) else success_sum/total_sum, "success_count:",
      success_sum, "cancer_count:", total_sum)

print("General===", "success_rate:", success_G_sum/maxn, "success_count:",
      success_G_sum, "test_count:", maxn)
print(predict_cancer_sum, total_sum)
