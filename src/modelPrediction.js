// src/modelPrediction.js

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { preprocessCustomers } = require('./dataProcessing');

/**
 * Load the trained TensorFlow.js model.
 */
let model = null;
async function loadModel() {
  if (!model) {
    try {
      model = await tf.loadLayersModel(
        `file://${path.join(__dirname, '../models/my_model/model.json')}`
      );
      console.log('Model loaded successfully.');
    } catch (error) {
      console.error('Error loading model:', error);
    }
  }
  return model;
}

/**
 * Load normalization data.
 */
function loadNormalizationData() {
  const normalizationPath = path.join(
    __dirname,
    '../models/my_model/normalizationData.json'
  );
  if (fs.existsSync(normalizationPath)) {
    const data = fs.readFileSync(normalizationPath, 'utf8');
    return JSON.parse(data);
  } else {
    console.error('Normalization data not found.');
    return null;
  }
}

/**
 * Get product recommendations for a given customer.
 * @param {Object} customer - Raw customer object.
 * @param {Array} products - Array of product objects.
 * @returns {Array} Array of all products ranked with scores and explanations.
 */
async function getRecommendations(customer, products) {
  await loadModel();

  // Preprocess the customer data
  const preprocessedCustomer = preprocessCustomers([customer])[0];

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

  // Extract features
  const inputContinuous = continuousFeatures.map(
    (feature) => preprocessedCustomer[feature] || 0
  );
  const inputCategorical = categoricalFeatures.map(
    (feature) => preprocessedCustomer[feature] || 0
  );

  // Load normalization data
  const normalizationData = loadNormalizationData();
  if (!normalizationData) {
    console.error('Normalization data is missing.');
    return [];
  }

  // Convert to tensor and normalize continuous inputs
  const inputTensor = tf.tensor2d([inputContinuous]);
  const inputMax = tf.tensor2d([normalizationData.inputMax]);
  const inputMin = tf.tensor2d([normalizationData.inputMin]);

  const denom = inputMax.sub(inputMin);
  const isZeroDenom = denom.equal(0);
  const safeDenom = denom.add(isZeroDenom.mul(1e-8));

  const normalizedContinuousInput = inputTensor.sub(inputMin).div(safeDenom);

  // Combine normalized continuous features with categorical features
  const categoricalTensor = tf.tensor2d([inputCategorical]);
  const normalizedInput = normalizedContinuousInput.concat(categoricalTensor, 1);

  // Make prediction
  const prediction = model.predict(normalizedInput);
  const predictionArray = prediction.arraySync()[0];

  // Create a list of products with scores
  const rankedProducts = products.map((product, index) => ({
    rank: index + 1, // Temporary rank, will be updated after sorting
    productName: product.name,
    score: predictionArray[index],
    features: product.features,
    price: product.price,
  }));

  // Sort the products based on the prediction scores in descending order
  rankedProducts.sort((a, b) => b.score - a.score);

  // Update rank after sorting
  rankedProducts.forEach((product, index) => {
    product.rank = index + 1;
  });

  // Generate explanations for each product
  const rankedProductsWithExplanations = rankedProducts.map((product) => {
    const explanation = generateAIExplanation(preprocessedCustomer, product);
    return {
      rank: product.rank,
      productName: product.productName,
      score: product.score,
      price: product.price,
      explanation: explanation,
    };
  });

  // Debugging Logs
  console.log('Input features:', inputContinuous.concat(inputCategorical));
  console.log('Normalized input:', normalizedInput.arraySync());
  console.log('Model prediction:', predictionArray);
  console.log('Top 3 Recommended products:');
  rankedProductsWithExplanations.slice(0, 3).forEach((prod) => {
    console.log(
      `${prod.rank}: ${prod.productName} | Score: ${prod.score.toFixed(
        4
      )} | Price: ${prod.price}`
    );
  });

  return rankedProductsWithExplanations;
}

/**
 * Generate explanations for a recommended product based on customer data and product features.
 * @param {Object} customer - Preprocessed customer data.
 * @param {Object} product - Product object.
 * @returns {String} Generated explanation.
 */
function generateAIExplanation(customer, product) {
  // Extract customer details
  const name = customer.customerName || 'Valued Customer';
  const devices = customer.total_devices;
  const avgBandwidth = (customer.avg_bandwidth_usage / 1e6).toFixed(2); // Convert to Mbps
  const coverageSize = customer.coverage_size;
  const networkSpeed = customer.network_speed;
  const state = getState(customer);
  const rssiMean = customer.rssi_mean;
  const rssiMin = customer.rssi_min;

  // Initialize explanation
  let explanation = `Hi ${name}, based on your current setup, we recommend ${product.productName}.`;

  // Customize explanation based on product
  if (product.productName.toLowerCase().includes('fiber')) {
    explanation += ` With your network speed of ${networkSpeed} Mbps and average bandwidth usage of ${avgBandwidth} Mbps, upgrading to ${product.productName} can provide you with faster and more reliable internet connectivity.`;
  } else if (product.productName.toLowerCase().includes('whole-home wi-fi')) {
    explanation += ` Considering you have ${devices} devices and a ${coverageSize} coverage area, ${product.productName} can help eliminate dead zones and ensure seamless connectivity throughout your home.`;
  } else if (product.productName.toLowerCase().includes('wi-fi security plus')) {
    explanation += ` Enhancing your security with ${product.productName} ensures comprehensive protection for all your devices, safeguarding against potential threats and vulnerabilities.`;
  } else if (product.productName.toLowerCase().includes('wi-fi security')) {
    explanation += ` To protect your network from malicious sites and potential cyber threats, ${product.productName} offers advanced security features tailored to your needs.`;
  } else if (product.productName.toLowerCase().includes('total shield')) {
    explanation += ` Given your high bandwidth usage and the need for robust security measures, ${product.productName} provides comprehensive protection for your network and devices.`;
  } else if (product.productName.toLowerCase().includes('my premium tech pro')) {
    explanation += ` Managing ${devices} devices can be challenging. ${product.productName} offers premium technical support to ensure all your devices run smoothly and efficiently.`;
  } else if (product.productName.toLowerCase().includes('identity protection')) {
    explanation += ` Living in ${state}, ${product.productName} helps safeguard your personal information, ensuring your data remains secure and protected from unauthorized access.`;
  } else if (product.productName.toLowerCase().includes('youtube tv')) {
    explanation += ` With your streaming needs and network capabilities, ${product.productName} offers a seamless TV experience without the need for additional hardware.`;
  } else if (product.productName.toLowerCase().includes('additional extender')) {
    explanation += ` To enhance your Wi-Fi coverage, ${product.productName} can help eliminate dead zones and ensure a strong signal throughout your property.`;
  } else if (product.productName.toLowerCase().includes('battery back-up for unbreakable wi-fi')) {
    explanation += ` Ensure uninterrupted internet access during outages with ${product.productName}, providing reliable backup power for your network.`;
  } else if (product.productName.toLowerCase().includes('unbreakable wi-fi')) {
    explanation += ` Maintain a stable internet connection even during unexpected outages with ${product.productName}, ensuring your connectivity remains uninterrupted.`;
  } else {
    explanation += ` ${product.features.join(' ')}. Consider adding it to enhance your internet experience.`;
  }

  return explanation;
}

/**
 * Get the state of the customer based on one-hot encoded state features.
 * @param {Object} customer - Preprocessed customer data.
 * @returns {String} State abbreviation or 'Unknown'.
 */
function getState(customer) {
  const states = ['TX', 'CA', 'CT', 'FL', 'OH', 'IN', 'PA', 'WV', 'IL', 'NV'];
  for (const state of states) {
    if (customer[`state_${state}`] === 1) {
      return state;
    }
  }
  return 'Unknown';
}

module.exports = { getRecommendations };