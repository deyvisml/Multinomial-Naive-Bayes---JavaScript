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

function calculateAccuracy(array1, array2) {
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
}

const predictDataset = (classifier, samples, y_test) => {
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
      result.push(res);
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
  const dataset = loadJSON("VOf1Dc_nxCA.json");
  const training = dataset.training;
  const testing = dataset.testing;

  console.log(
    "Num. class 0 samples (training): ",
    training.classes.filter((valor) => valor == 0).length
  );
  console.log(
    "Num. class 1 samples (training): ",
    training.classes.filter((valor) => valor == 1).length
  );

  const X_train = preprocessComments(training.comments);
  const y_train = training.classes;

  const classifier = new MultinomialNB();

  classifier.fit(X_train, y_train);

  const X_test = preprocessComments(testing.comments);
  const y_test = testing.classes;

  /*
  const y_pred = classifier.predict(X_test);

  //calculating accuracy
  const accuracy = calculateAccuracy(y_pred, y_test);

  console.log("Accuracy:", accuracy);

  predictDataset(classifier, testing.comments, y_test);

  classifier.save("outputs/model.json");
  */
  const ss = classifier.predict(
    preprocessComments(["Ese es su contacto de WhatsApp 👆👆"])
  );
  console.log("debug:", ss);
  return true;
};

main();
