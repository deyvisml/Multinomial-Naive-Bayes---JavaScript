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

/**
 * Allows to test and evaluated a dataset through a multnomial nb model
 * @param {function} classifier Allows make clasifications
 * @param {array} comments Dataset to test the model
 * @param {array} classes Real classes of the samples
 * @param {bool} wrong_only To only save when the model makes a wrong prediction
 * @param {string} real_class When wrong only is true, then you could want not all the wrong prections, e.g. only get the wrong result when the real class was spam (1)
 */
const predictDataset = (
  classifier,
  comments,
  classes,
  wrong_only = false,
  real_class = undefined
) => {
  const tokens = preprocessComments(comments);
  const y_pred = classifier.predict(tokens);

  const result = [];

  for (let i = 0; i < comments.length; i++) {
    const res = {};
    res["comment"] = comments[i];
    res["tokens"] = tokens[i];
    res["real"] = classes[i];
    res["pred"] = y_pred[i];

    if (wrong_only) {
      if (res["real"] != res["pred"]) {
        if (real_class) {
          if (res["real"] == real_class) {
            result.push(res);
          }
        } else {
          result.push(res);
        }
      }
    } else {
      result.push(res);
    }
  }

  saveJSON("pred_samples.json", { result: result });
};

// Main to generate a model that will be use in the web extension
const main = () => {
  let dataset = loadJSON("dataset_no_duplicates.json");
  let { training, testing } = dataset;

  console.log(
    "Num Spam samples (training 80%)\t:",
    training.y.filter((valor) => valor == 1).length
  );
  console.log(
    "Num Ham samples (training 80%)\t:",
    training.y.filter((valor) => valor == 0).length
  );

  // ======= TRAINING =======

  const X = preprocessComments(training.x);
  const y = training.y;

  const classifier = new MultinomialNB();

  classifier.fit(X, y);

  // ======= TESTING =======

  // test for a simple sample
  /*
  const sample =
    "	Bonito e informativo vídeo. centrémonos siempre en cómo ganar, solía ver el comercio de criptomonedas como algo secundario, pero resultó ser una fuente importante de ingresos pasivos desde que conocí al Sr. Harry Martins, su experiencia en el mercado de criptomonedas es insuperable";
  classifier.predict(preprocessComments([sample]));
  return;*/

  const X_test = preprocessComments(testing.x);
  const y_test = testing.y;

  const y_pred = classifier.predict(X_test);

  // calculating accuracy
  const accuracy = calculateAccuracy(y_pred, y_test);
  console.log("Accuracy:", accuracy);

  // calculating confussion matrix
  const { TP, FN, FP, TN } = calculateConfussionMatrix(y_pred, y_test);
  console.log("Confussion Matrix:");
  console.log("-> TP:", TP);
  console.log("-> FN:", FN);
  console.log("-> FP (*):", FP);
  console.log("-> TN:", TN);

  let EVALUATE_DATA = false;
  if (EVALUATE_DATA) {
    predictDataset(
      classifier,
      testing.x,
      testing.y,
      (wrong_only = true),
      (real_class = "1")
    );
  }

  classifier.save("outputs/model.json");

  return true;
};

/*
// Main for training and testing
const main = () => {
  let dataset = loadJSON("dataset3.json");

  let { training, testing } = dataset;

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

  //dataset = loadJSON("05_2ibqfxEAESo.json");
  //testing = training;

  // test for a simple sample
 

  const X_test = preprocessComments(testing.comments);
  const y_test = testing.classes;

  const y_pred = classifier.predict(X_test);

  //calculating accuracy
  const accuracy = calculateAccuracy(y_pred, y_test);
  console.log("Accuracy:", accuracy);

  //calculating confussion matrix
  const { TP, FN, FP, TN } = calculateConfussionMatrix(y_pred, y_test);
  console.log("Confussion Matrix:");
  console.log("-> TP:", TP);
  console.log("-> FN:", FN);
  console.log("-> FP (*):", FP);
  console.log("-> TN:", TN);

  predictDataset(
    classifier,
    testing.comments,
    testing.classes,
    (wrong_only = true),
    (real_class = "1")
  );

  classifier.save("outputs/model.json");

  return true;
};*/

main();
