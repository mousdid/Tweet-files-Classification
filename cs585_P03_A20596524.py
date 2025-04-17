import sys
from cs585_P03_A20596524_EXTRA import loader, Classifier, Vectorizer, evaluate_metrics
from joblib import dump, load
import os

def main():

    if len(sys.argv) == 3:
        try:
            algo = int(sys.argv[1])
            train_size = int(sys.argv[2])
        except ValueError:
            algo, train_size = 0, 80
        if algo not in [0, 1] or not (50 <= train_size <= 80):
            algo, train_size = 0, 80
    else:
        algo, train_size = 0, 80

    print("\nOusdid, Mohamed Yassir, A20596524 solution:")
    print(f"Training set size: {train_size}%")
    print("Classifier type:", "Naive Bayes" if algo == 0 else "Logistic Regression")
    model_name = "nb" if algo == 0 else "lr"
    filename   = f"classifier_{model_name}_{train_size}.joblib"
    TRAIN_SIZE = train_size / 100.0
    path = 'Dataset/'
    ld = loader(path)
    df = ld.load_data()
    df = ld.preprocess_data()

    train_df, test_df = ld.split_data(TRAIN_SIZE)


    #to not rerun training
    if os.path.exists(filename):
        vectorizer,clf = load(filename)


    else:   
        vectorizer = Vectorizer(train_df['text'])
        X_train = vectorizer.transform_batch(train_df['text'])
        y_train = list(train_df['male'])
        clf = Classifier(algo, TRAIN_SIZE, vocab=vectorizer.vocab)
        print("\nTraining classifier...")
        clf.train(X_train, y_train)
        dump((vectorizer, clf), filename)



    print("\nTesting classifier...")
    X_test = vectorizer.transform_batch(test_df['text'])
    y_test = list(test_df['male'])
    y_pred = [clf.predict_vector(x)[0] for x in X_test]
    evaluate_metrics(y_test, y_pred)
    



    while True:
        sent = input("\nEnter your sentence/document:\n").strip().lower()
        if not sent:
            break
        vec = vectorizer.transform(sent)
        pred, probs = clf.predict_vector(vec)
        label = "True" if pred else "False"

        print(f"\nSentence/document S: {sent}\nwas classified as {label}.")
        if algo == 0:
            print(f"P(True | S) = {probs[True]:.4f}")
            print(f"P(False | S) = {probs[False]:.4f}")

        again = input("\nDo you want to enter another sentence [Y/N]? ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main()

#demo:
#football is awesome
#i like shopping 
#makeup is life
