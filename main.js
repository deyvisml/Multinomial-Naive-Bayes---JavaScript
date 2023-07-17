const fs = require("fs");
const MultinomialNB = require("./MultinomialNB");
const text_preprocessor = require("./TextPreprocessor");

const preprocessComments = (comments) => {
  const preprocessed_comments = [];
  for (const comment of comments) {
    preprocessed_comments.push(text_preprocessor.preprocess(comment));
  }

  return preprocessed_comments;
};

const loadJSON = (filename) => {
  let dataset = fs.readFileSync("./datasets/" + filename, "utf-8");
  dataset = JSON.parse(dataset);

  return dataset;
};

const saveJSON = (filename, data) => {
  const json_data = JSON.stringify(data);

  try {
    fs.writeFileSync("outputs/" + filename, json_data, "utf-8");
    console.log("File saved successfully.");
  } catch (error) {
    console.error("Error to write the file:", error);
  }
};

const calculateAccuracy = (array1, array2) => {
  if (array1.length !== array2.length) {
    throw new Error("The arrays must have the same length");
  }

  let counter = 0;
  for (let i = 0; i < array1.length; i++) {
    if (array1[i] == array2[i]) {
      counter++;
    }
  }

  const accuracy = counter / array1.length;
  return accuracy;
};

const calculateConfussionMatrix = (y_pred, y_test) => {
  let TP = 0;
  let FN = 0;
  let FP = 0;
  let TN = 0;

  if (y_pred.length !== y_test.length) {
    throw new Error("The arrays must have the same length");
  }

  for (let i = 0; i < y_test.length; i++) {
    if (y_test[i] == 1) {
      if (y_pred[i] == 1) TP++;
      else FN++;
    } else {
      if (y_pred[i] == 1) FP++;
      else TN++;
    }
  }

  return { TP, FN, FP, TN };
};

const predictDataset = (classifier, samples, y_test, wrong_only = false) => {
  const tokens = preprocessComments(samples);
  const y_pred = classifier.predict(tokens);

  const result = [];

  for (let i = 0; i < samples.length; i++) {
    if (y_test) {
      const res = {};
      res["comment"] = samples[i];
      res["tokens"] = tokens[i];
      res["real"] = y_test[i];
      res["pred"] = y_pred[i];

      if (wrong_only) {
        if (res["real"] != res["pred"]) {
          result.push(res);
        }
      } else {
        result.push(res);
      }
    } else if (y_pred[i] == 1) {
      const res = {};
      res["comment"] = samples[i];
      res["pred"] = y_pred[i];
      result.push(res);
    }
  }

  saveJSON("pred_samples.json", { result: result });
};

const main = () => {
  let dataset = loadJSON("dataset.json");
  let training = dataset.training;
  let testing = dataset.testing;

  console.log(
    "Num. class 0 samples (training): ",
    training.classes.filter((valor) => valor == 0).length
  );
  console.log(
    "Num. class 1 samples (training): ",
    training.classes.filter((valor) => valor == 1).length
  );

  // ======= TRAINING =======

  const X_train = preprocessComments(training.comments);
  const y_train = training.classes;

  const classifier = new MultinomialNB();

  classifier.fit(X_train, y_train);

  // ======= TESTING =======
  dataset = loadJSON("TYfQZA4ZaXs.json");
  testing = dataset.all;

  const X_test = preprocessComments(testing.comments);
  const y_test = testing.classes;

  const y_pred = classifier.predict(X_test);

  //calculating accuracy
  const accuracy = calculateAccuracy(y_pred, y_test);
  console.log("Accuracy:", accuracy);

  //calculating confussion matrix
  const confussion_matrix = calculateConfussionMatrix(y_pred, y_test);
  console.log("Confussion Matrix:");
  console.log("-> TP:", confussion_matrix.TP);
  console.log("-> FN:", confussion_matrix.FN);
  console.log("-> FP (*):", confussion_matrix.FP);
  console.log("-> TN:", confussion_matrix.TN);

  predictDataset(classifier, testing.comments, y_test, (wrong_only = true));

  classifier.save("outputs/model.json");

  return true;
};

main();
