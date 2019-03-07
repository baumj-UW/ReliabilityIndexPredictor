# ReliabilityIndexPredictor
SAIDI and SAIFI Predictor based on public utility data 

The goal of this project is to develop a regression model that predicts the annual System Average
Interruption Duration Index (SAIDI) and System Average Interruption Frequency Index (SAIFI) of
different utilities across the United States based on data published by the Energy Information
Administration (EIA). As the index names suggest, SAIDI and SAIFI are key metrics of a utility’s
reliability performance and their forecasted values could be useful for system planning and evaluation.

To generate a model, data from the “Electric power sales, revenue, and efficiency Form EIA-861”
collection of utility statistics from 2012 through 2017 will be aggregated and processed. The Form EIA-
861 surveys over 1,000 utilities in the U.S. on operational information such as region, distribution
management, and energy market interaction, and has reported annual SAIDI and SAIFI information since
2013 [1]. The data will be split by year for training and test; 2012-2014 will be used for training, 2015
for tuning and validation, and 2016-2017 for test. Initial investigation of this data source was influenced
by an Inside Energy article [2] which summarized the SAIDI and SAIFI statistics after their first year of
publication.

[1] U.S. Energy Information Administration, "Electric power sales, revenue, and energy efficiency Form EIA-861
detailed data files," 2019. [Online]. Available: https://www.eia.gov/electricity/data/eia861/.
'''
