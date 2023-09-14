const fs = require("fs");

class MultinomialNB {
  constructor(model) {
    this.laplace_smoothing = 1; // constant, no change

    if (model) {
      this.frequency = model.frequency;
      this.class_names = model.class_names;
      this.vocabulary = model.vocabulary;
      this.log_prior = model.log_prior;
      this.log_likelihood = model.log_likelihood;
    } else {
      this.frequency = {}; // contains words (also objects) and inside each word, the frecuency of that word in each class
      this.class_names = [];
      this.vocabulary = [];
      this.log_prior = {}; // to save log_prior of each class
      this.log_likelihood = {}; // la key prodria ser una palabra y adentro un objeto con las clases
    }
  }

  /* X sera un arrays de arrays, e y solo sera un array, ambos ya deben haber sido preprocesados */
  fit(X, y) {
    if (X.length != y.length)
      console.log("Error, the length of X and y are diferent.");

    // convert all the classes to string values (even if there are already strings)
    y = y.map(function (numero) {
      return numero.toString();
    });

    this.class_names = [...new Set(y)];
    const num_documents = X.length;

    // calculating frequency and vocabulary
    for (const [index, y_i] of y.entries()) {
      for (const word of X[index]) {
        if (!this.vocabulary.includes(word)) {
          this.vocabulary.push(word);
        }
        this.frequency[word] = this.frequency[word] ?? {};
        this.frequency[word][y_i] = (this.frequency[word]?.[y_i] ?? 0) + 1;
      }
    }

    for (const class_name of this.class_names) {
      const num_documents_class_name = y.filter(
        (item) => item === class_name
      ).length;

      this.log_prior[class_name] = Math.log(
        num_documents_class_name / num_documents
      );

      // calculating the denominator of log_likelihood
      let denominator = 0;
      for (const word of this.vocabulary) {
        denominator += this.frequency[word]?.[class_name] ?? 0;
      }
      denominator += this.vocabulary.length * this.laplace_smoothing;

      for (const word of this.vocabulary) {
        this.log_likelihood[word] = this.log_likelihood[word] ?? {};
        this.log_likelihood[word][class_name] =
          ((this.frequency[word]?.[class_name] ?? 0) + this.laplace_smoothing) /
          denominator;
      }
    }
  }

  predict(X) {
    const preds = [];

    for (const x of X) {
      let pred = null;
      const scores = {};

      for (const class_name of this.class_names) {
        scores[class_name] = 0; //this.log_prior[class_name]; // log prior is generating incorrect clasification, so it appear that i need a a uniform distribution of data 50%, 50%

        for (const word of x) {
          if (this.vocabulary.includes(word)) {
            scores[class_name] += this.log_likelihood[word][class_name];
            /*
            console.log(
              "class, word, likelihood =>",
              class_name,
              word,
              "\t",
              this.log_likelihood[word][class_name]
            );*/
          }
        }
      }
      //console.log("scores -> ", scores);
      /*
      console.log(
        "scores -> ",
        scores,
        scores["1"] - scores["0"],
        scores["1"] - scores["0"] > -1.704
      );*/

      const max_probability = Math.max(...Object.values(scores));
      pred = Object.keys(scores).find((key) => scores[key] === max_probability);

      // modifying: reduce false positive rate

      let threshold = 1.2; // good value: 1.45 (umbral) 1 is default, but a good value is around 1.4 (and 1.8 for testing with TYfQZA4ZaXs)
      // la segunda parte de la condicional es agregada para lidiar con comentarios largos (varias palabras)
      pred = scores["1"] / scores["0"] >= threshold ? "1" : "0";

      preds.push(pred);
    }

    return preds;
  }

  /**
   * Save the current model
   * @param {string} filename File name (.json) to save the model
   */
  save(filename) {
    const model = {
      name: "MultinomialNB",
      frequency: this.frequency,
      class_names: this.class_names,
      vocabulary: this.vocabulary,
      log_prior: this.log_prior,
      log_likelihood: this.log_likelihood,
    };
    const modelJSON = JSON.stringify(model);

    try {
      fs.writeFileSync(filename, modelJSON, "utf-8");
      console.log("File saved successfully.");
    } catch (error) {
      console.error("Error to write the file:", error);
    }
  }

  /**
   * Creates a new MBN model fron the givin json file, static method, I can be call without any instance
   * @param {string} filename Filename of model in .json format
   * @returns {MBN}
   */
  static load(filename) {
    let model;

    // https://stackoverflow.com/a/14078644
    try {
      const data = fs.readFileSync(filename, "utf-8");
      model = JSON.parse(data);
    } catch (error) {
      console.error("Error reading the file:", error);
    }

    return new MultinomialNB(model);
  }
}

module.exports = MultinomialNB;
