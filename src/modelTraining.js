// src/modelTraining.js

const tf = require('@tensorflow/tfjs-node');
const {
  loadCustomerData,
  loadProductData,
  preprocessCustomers,
} = require('./dataProcessing');
const path = require('path');
const fs = require('fs');

async function trainModel() {
  try {
    console.log('Starting model training...');

    // Load and preprocess data
    console.log('Loading customer data...');
    const customersRaw = await loadCustomerData();
    console.log(`Loaded ${customersRaw.length} raw customer records.`);

    console.log('Loading product data...');
    const products = loadProductData();
    console.log(`Loaded ${products.length} products.`);

    // Check if products are loaded
    if (products.length === 0) {
      console.error(
        'No products found. Please check your product_catalog.md file.'
      );
      return;
    }

    console.log('Preprocessing customer data...');
    const customers = preprocessCustomers(customersRaw);
    console.log(`Preprocessed ${customers.length} customer records.`);

    const numProducts = products.length;
    console.log(`Number of products: ${numProducts}`);

    // Define feature names
    const continuousFeatures = [
      'total_devices',
      'avg_bandwidth_usage',
      'network_speed',
      'tx_avg_bps',
      'rx_p95_bps',
      'tx_p95_bps',
      'rx_max_bps',
      'tx_max_bps',
      'rssi_mean',
      'rssi_median',
      'rssi_max',
      'rssi_min',
    ];
    const categoricalFeatures = [
      'coverage_size_Small',
      'coverage_size_Medium',
      'coverage_size_Large',
      'state_TX',
      'state_CA',
      'state_CT',
      'state_FL',
      'state_OH',
      'state_IN',
      'state_PA',
      'state_WV',
      'state_IL',
      'state_NV',
      // Add more states if necessary
    ];

    console.log('Extracting continuous features...');
    const continuousInputs = customers.map((customer) =>
      continuousFeatures.map((feature) => customer[feature] || 0)
    );

    console.log('Extracting categorical features...');
    const categoricalInputs = customers.map((customer) =>
      categoricalFeatures.map((feature) => customer[feature] || 0)
    );

    console.log('Converting inputs to tensors...');
    const continuousTensor = tf.tensor2d(continuousInputs);
    const categoricalTensor = tf.tensor2d(categoricalInputs);

    console.log('Normalizing continuous features...');
    const inputMax = continuousTensor.max(0);
    const inputMin = continuousTensor.min(0);
    const denom = inputMax.sub(inputMin);
    const isZeroDenom = denom.equal(0);
    const safeDenom = denom.add(isZeroDenom.mul(1e-8));

    const normalizedContinuousInputs = continuousTensor
      .sub(inputMin)
      .div(safeDenom);

    console.log('Combining normalized continuous and categorical features...');
    const normalizedInputs = normalizedContinuousInputs.concat(
      categoricalTensor,
      1
    );

    // Save normalization parameters
    console.log('Saving normalization data...');
    const normalizationData = {
      inputMax: inputMax.arraySync(),
      inputMin: inputMin.arraySync(),
    };

    const modelDir = path.join(__dirname, '../models/my_model');
    if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true });
      console.log('Created models/my_model directory.');
    }

    fs.writeFileSync(
      path.join(modelDir, 'normalizationData.json'),
      JSON.stringify(normalizationData)
    );
    console.log('Normalization data saved.');

    // Define labels based on customer features
    console.log('Assigning labels based on customer data...');
    const labels = customers.map((customer, idx) => {
      const label = Array(numProducts).fill(0);

      // Fiber Speed Tiers
      if (customer.network_speed <= 500) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'fiber 500'
        );
        if (index >= 0) label[index] = 1;
      } else if (customer.network_speed <= 1000) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'fiber 1 gig'
        );
        if (index >= 0) label[index] = 1;
      } else if (customer.network_speed <= 2000) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'fiber 2 gig'
        );
        if (index >= 0) label[index] = 1;
      } else if (customer.network_speed <= 5000) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'fiber 5 gig'
        );
        if (index >= 0) label[index] = 1;
      } else {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'fiber 7 gig'
        );
        if (index >= 0) label[index] = 1;
      }

      // Whole-Home Wi-Fi
      if (customer.coverage_size_Large === 1 || customer.rssi_min < -80) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'whole-home wi-fi'
        );
        if (index >= 0) label[index] = 1;
      }

      // Wi-Fi Security
      if (
        !customer.wifi_security &&
        !customer.wifi_security_plus &&
        !customer.total_shield &&
        customer.avg_bandwidth_usage > 1000000
      ) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'wi-fi security'
        );
        if (index >= 0) label[index] = 1;
      }

      // Wi-Fi Security Plus
      if (customer.wifi_security_plus) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'wi-fi security plus'
        );
        if (index >= 0) label[index] = 1;
      }

      // Total Shield
      if (customer.avg_bandwidth_usage > 5000000) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'total shield'
        );
        if (index >= 0) label[index] = 1;
      }

      // My Premium Tech Pro
      if (customer.total_devices > 15 || customer.rssi_mean < -70) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'my premium tech pro'
        );
        if (index >= 0) label[index] = 1;
      }

      // Identity Protection
      if (customer.state_CA === 1 || customer.state_TX === 1) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'identity protection'
        );
        if (index >= 0) label[index] = 1;
      }

      // YouTube TV
      if (customer.rx_avg_bps > 10000000) {
        const index = products.findIndex(
          (p) => p.name.toLowerCase() === 'youtube tv'
        );
        if (index >= 0) label[index] = 1;
      }

      // Additional rules can be added here...

      return label;
    });

    console.log('Labels assigned.');

    // **New Code: Inspect Labels**
    const sampleSize = 10; // Adjust as needed
    console.log('\n--- Label Samples ---');
    for (let i = 0; i < Math.min(sampleSize, customers.length); i++) {
      console.log(`Customer: ${customers[i].customerName}`);
      console.log('Labels:');
      labels[i].forEach((label, idx) => {
        if (label === 1) {
          console.log(`  - ${products[idx].name}`);
        }
      });
      console.log('----------------------');
    }
    console.log('--- End of Label Samples ---\n');

    // **New Code: Analyze Label Distribution**
    const labelCounts = {};
    products.forEach((product) => {
      labelCounts[product.name] = 0;
    });

    labels.forEach((label) => {
      label.forEach((val, idx) => {
        if (val === 1) {
          labelCounts[products[idx].name]++;
        }
      });
    });

    console.log('\n--- Label Distribution ---');
    for (const [product, count] of Object.entries(labelCounts)) {
      console.log(`${product}: ${count}`);
    }
    console.log('--- End of Label Distribution ---\n');

    console.log('Converting labels to tensor...');
    const labelTensor = tf.tensor2d(labels);
    console.log('Labels tensor created.');

    // Define the model
    console.log('Defining the model architecture...');
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [normalizedInputs.shape[1]],
        units: 128, // Increased units for better learning
        activation: 'relu',
      })
    );
    model.add(tf.layers.dropout({ rate: 0.3 })); // Dropout layer to prevent overfitting
    model.add(tf.layers.dense({ units: 64, activation: 'relu' })); // Additional hidden layer
    model.add(tf.layers.dropout({ rate: 0.3 })); // Another dropout layer
    model.add(tf.layers.dense({ units: 32, activation: 'relu' })); // Further hidden layer
    model.add(
      tf.layers.dense({ units: numProducts, activation: 'sigmoid' })
    ); // Output layer for multi-label classification

    console.log('Compiling the model...');
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy', // Suitable for multi-label classification
      metrics: ['accuracy'],
    });

    console.log('Starting model training...');
    const history = await model.fit(normalizedInputs, labelTensor, {
      epochs: 150, // Increased epochs for better learning
      batchSize: 16,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(
              4
            )}, val_loss = ${logs.val_loss.toFixed(
              4
            )}, accuracy = ${logs.acc.toFixed(
              4
            )}, val_accuracy = ${logs.val_acc.toFixed(4)}`
          );
        },
        onTrainEnd: () => {
          console.log('Training completed.');
        },
      },
    });

    console.log('Model training complete.');

    // Save the model
    console.log('Saving the trained model...');
    await model.save(`file://${modelDir}`);
    console.log('Model saved successfully.');
  } catch (error) {
    console.error('Error during model training:', error);
  }
}

trainModel();