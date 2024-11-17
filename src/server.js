// src/server.js

const express = require('express');
const app = express();
const path = require('path');
const { loadCustomerData, loadProductData } = require('./dataProcessing');
const { getRecommendations } = require('./modelPrediction');

const PORT = 3000;

let customers = [];
let products = [];

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, '../public')));

// Load data on server start
(async () => {
  try {
    console.log('Loading customer data...');
    customers = await loadCustomerData();
    console.log(`Loaded ${customers.length} customers.`);

    console.log('Loading product data...');
    products = loadProductData();
    console.log(`Loaded ${products.length} products.`);
    products.forEach((product, index) => {
      console.log(`${index}: ${product.name}`);
    });

    if (products.length === 0) {
      console.error(
        'No products found. Please check your product_catalog.md file.'
      );
      process.exit(1); // Exit the process if no products are loaded
    }

    console.log('Server is running at http://localhost:' + PORT);
    console.log('Sample customer data:', customers[0]);
    console.log(
      'Loaded customer names:',
      customers.map((c) => c.customerName)
    );
  } catch (error) {
    console.error('Error loading data:', error);
    process.exit(1); // Exit the process if there's an error loading data
  }
})();

/**
 * API Endpoint: GET /recommendations
 * Query Parameters:
 *  - customerName: Name of the customer to get recommendations for.
 */
app.get('/recommendations', async (req, res) => {
  console.log('Received GET /recommendations');
  try {
    const customerName = req.query.customerName;
    console.log('Searching for customer:', customerName);
    if (!customerName) {
      return res.status(400).send('Customer name is required.');
    }

    const customer = customers.find(
      (c) =>
        c.customerName &&
        c.customerName.toLowerCase() === customerName.toLowerCase()
    );

    if (!customer) {
      return res.status(404).send('Customer not found.');
    }

    const rankedRecommendations = await getRecommendations(customer, products);

    res.json(rankedRecommendations);
  } catch (error) {
    console.error('Error getting recommendations:', error);
    res.status(500).send('Internal server error.');
  }
});

// Start the server
app.listen(PORT);