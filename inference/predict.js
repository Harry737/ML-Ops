const fs = require("fs");
const axios = require("axios");

async function predict() {
  const img = fs.readFileSync("mnist_test_images/7_label_9.png");

  // convert image to base64 (V2 requires JSON)
  const base64 = img.toString("base64");

  const payload = {
    inputs: [
      {
        name: "input-0",
        shape: [1],
        datatype: "BYTES",
        data: [base64]
      }
    ]
  };

  const res = await axios.post(
    "http://localhost:8081/v1/models/mnist:predict",
    payload,
    { headers: { "Content-Type": "application/json" } }
  );

  console.log("Prediction:", JSON.stringify(res.data, null, 2));
}

predict().catch(console.error);
