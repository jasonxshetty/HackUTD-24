// src/dataProcessing.js

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

/**
 * Load customer data from CSV.
 * @returns {Promise<Array>} Array of customer objects.
 */
async function loadCustomerData() {
  return new Promise((resolve, reject) => {
    const customers = [];
    fs.createReadStream(path.join(__dirname, '../data/current_customers.csv'))
      .pipe(csv())
      .on('data', (data) => {
        customers.push(data);
      })
      .on('end', () => {
        // Convert property names to match your code's expectations
        const processedCustomers = customers.map((customer) => {
          // Calculate total_devices
          const wirelessClients = parseInt(customer.wireless_clients_count) || 0;
          const wiredClients = parseInt(customer.wired_clients_count) || 0;
          const totalDevices = wirelessClients + wiredClients;

          // Parse avg_bandwidth_usage
          const avgBandwidthUsage =
            parseFloat(customer.rx_avg_bps) || 0;

          // Extract numeric value from network_speed (e.g., '500.0M' to 500)
          let networkSpeed = 0;
          if (customer.network_speed) {
            const speedMatch = customer.network_speed.match(/([\d\.]+)([A-Za-z]+)/);
            if (speedMatch) {
              const speedValue = parseFloat(speedMatch[1]);
              const speedUnit = speedMatch[2].toUpperCase();
              if (speedUnit === 'M') {
                networkSpeed = speedValue;
              } else if (speedUnit === 'G') {
                networkSpeed = speedValue * 1000;
              } else {
                networkSpeed = speedValue;
              }
            }
          }

          // For coverage_size, set a default or infer from extenders or total_devices
          let coverageSize = 'Medium'; // Default value
          const extenders = parseInt(customer.extenders) || 0;
          if (extenders >= 2 || totalDevices > 15) {
            coverageSize = 'Large';
          } else if (extenders === 0 && totalDevices <= 5) {
            coverageSize = 'Small';
          }

          // Parse boolean fields
          const wifiSecurity =
            customer.wifi_security &&
            (customer.wifi_security.toLowerCase() === 'true' ||
              customer.wifi_security.toLowerCase() === '1')
              ? 1
              : 0;
          const wifiSecurityPlus =
            customer.wifi_security_plus &&
            (customer.wifi_security_plus.toLowerCase() === 'true' ||
              customer.wifi_security_plus.toLowerCase() === '1')
              ? 1
              : 0;
          const totalShield =
            customer.total_shield &&
            (customer.total_shield.toLowerCase() === 'true' ||
              customer.total_shield.toLowerCase() === '1')
              ? 1
              : 0;

          // One-hot encode 'state'
          const states = ['TX', 'CA', 'CT', 'FL', 'OH', 'IN', 'PA', 'WV', 'IL', 'NV'];
          const stateOneHot = {};
          states.forEach((state) => {
            stateOneHot[`state_${state}`] =
              customer.state && customer.state.toUpperCase() === state ? 1 : 0;
          });

          return {
            customerName:
              customer.CustomerName ||
              customer.customerName ||
              customer.customer_name ||
              '',
            total_devices: totalDevices,
            avg_bandwidth_usage: avgBandwidthUsage,
            network_speed: networkSpeed,
            coverage_size: coverageSize,
            wifi_security: wifiSecurity,
            wifi_security_plus: wifiSecurityPlus,
            total_shield: totalShield,
            rx_avg_bps:
              parseFloat(customer.rx_avg_bps) ||
              parseFloat(customer.rxAvgBps) ||
              0,
            // Additional numerical features
            tx_avg_bps: parseFloat(customer.tx_avg_bps) || 0,
            rx_p95_bps: parseFloat(customer.rx_p95_bps) || 0,
            tx_p95_bps: parseFloat(customer.tx_p95_bps) || 0,
            rx_max_bps: parseFloat(customer.rx_max_bps) || 0,
            tx_max_bps: parseFloat(customer.tx_max_bps) || 0,
            rssi_mean: parseFloat(customer.rssi_mean) || 0,
            rssi_median: parseFloat(customer.rssi_median) || 0,
            rssi_max: parseFloat(customer.rssi_max) || 0,
            rssi_min: parseFloat(customer.rssi_min) || 0,
            // One-hot encoded features
            ...stateOneHot,
            // Add any other relevant features as needed
          };
        });
        resolve(processedCustomers);
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

/**
 * Load product data from product_catalog.md (Markdown table format).
 * @returns {Array} Array of product objects.
 */
function loadProductData() {
  const productFilePath = path.join(__dirname, '../data/product_catalog.md');
  if (!fs.existsSync(productFilePath)) {
    console.error('product_catalog.md file not found in the data directory.');
    return [];
  }

  const fileContent = fs.readFileSync(productFilePath, 'utf8');

  // Split the content into lines
  const lines = fileContent.split('\n');

  // Initialize an array to hold products
  const products = [];

  // Flags to identify the table content
  let inTable = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();

    // Identify the start of the table
    if (line.startsWith('|') && line.includes('Product Name')) {
      inTable = true;
      // Skip the header line and the separator line
      i += 2;
      continue;
    }

    if (inTable && line.startsWith('|')) {
      const columns = line.split('|').map((col) => col.trim());
      // Remove the empty first and last elements due to leading and trailing '|'
      columns.shift();
      columns.pop();

      if (columns.length >= 3) {
        const productName = columns[0];
        const featuresText = columns[1];
        const priceText = columns[2];

        // Split features by periods or bullets
        const features = featuresText
          .split(/[\.\•]/)
          .map((feature) => feature.trim())
          .filter((feature) => feature !== '');

        // Extract price value
        const priceMatch = priceText.match(/\$([\d\.]+)/);
        const price = priceMatch ? parseFloat(priceMatch[1]) : 0;

        // Create product object
        const product = {
          name: productName,
          features: features,
          price: price,
        };

        products.push(product);
      }
    }
  }

  console.log(`Loaded ${products.length} products from product_catalog.md`);
  products.forEach((product, index) => {
    console.log(`${index}: ${product.name}`);
  });

  return products;
}

/**
 * Preprocess customers by one-hot encoding categorical features.
 * @param {Array} customers - Array of customer objects.
 * @returns {Array} Array of preprocessed customer objects.
 */
function preprocessCustomers(customers) {
  return customers.map((customer) => {
    // One-hot encode 'coverage_size'
    const coverageSizes = ['Small', 'Medium', 'Large'];
    const coverageSizeOneHot = {};
    coverageSizes.forEach((size) => {
      coverageSizeOneHot[`coverage_size_${size}`] =
        customer.coverage_size.toLowerCase() === size.toLowerCase() ? 1 : 0;
    });

    // One-hot encode 'state'
    const states = ['TX', 'CA', 'CT', 'FL', 'OH', 'IN', 'PA', 'WV', 'IL', 'NV'];
    const stateOneHot = {};
    states.forEach((state) => {
      stateOneHot[`state_${state}`] =
        customer[`state_${state}`] === 1 ? 1 : 0;
    });

    return {
      ...customer,
      ...coverageSizeOneHot,
      ...stateOneHot,
      // Add any other processed features as needed
    };
  });
}

module.exports = { loadCustomerData, loadProductData, preprocessCustomers };