#2024.04.08 + trendcharts
#2024.04.02.10 widelayot + tab stats historical data
#2024.04.01.23
#2024.04.01.18

import streamlit as st
import pandas as pd

#forecasting


from prophet import Prophet

import prophet as ph

from statsmodels.tsa.api import SimpleExpSmoothing

# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA


import plotly.express as px


# für Excel-Export-Funktionen
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


#option menu
from streamlit_option_menu import option_menu



#forecast mit linear regression
#from sklearn.linear_model import LinearRegression
import numpy as np


st.set_page_config(
    page_title="Simple Preditictions",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)



#---Option Menu -------------------

option = option_menu(
	menu_title="Simple Predictions",
	options=["Prophet", "Statsmodels","Trendchart"],
	icons=["0-square", "2-circle", "3-circle"], #https://icons.getbootstrap.com/
	orientation="horizontal",
)








#Code um den Button-Design anzupassen
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ce1126;
    color: white;
    height: 3em;
    width: 14em;
    border-radius:10px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
}
div.stButton > button:hover {
	background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
	background-color:#ce1126;
}
div.stButton > button:active {
	position:relative;
	top:3px;
}
</style>""", unsafe_allow_html=True)



#Variable definitions:

df = pd.DataFrame()



###File Uploader

data_file = st.sidebar.file_uploader("Upload an  file with your time series data", type=["xlsx", "csv"])

if data_file is not None:

    if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  # Excel file
        df = pd.read_excel(data_file)
    elif data_file.type == "text/csv":  # CSV file
        df = pd.read_csv(data_file)





if option =="Prophet":


    st.title("Time Series Forecasting with Prophet")

    if data_file is None:
        st.warning("No data loaded")

    if len (df)>0:

        # Show the first 5 rows of the dataframe
        st.write("Here are the first 5 rows of your data:")
        st.dataframe(df.head())

        # Let the user select the columns for the date and the target

        # Filter numerical columns
        numerical_cols = df.select_dtypes(include=[float, int]).columns.tolist()

        object_cols = df.select_dtypes(include='object').columns.tolist()

        st.write("")

        st.write(""" ### Select a date variable """)

        createDateTime = st.checkbox("Create a datetime variable from the index")
        if createDateTime:
            df["createdDateTime"] = pd.to_datetime(df.index+200,unit='W')
            #df = df.rename(columns={createDateTime: "ds"})
            df['ds'] = df['createdDateTime']

            #df['createdDateTime'] = df['createdDateTime'].dt.to_period('M')
            st.write("df mit createdDateTime: ",df)

            #df[date_col] = pd.to_datetime(df[date_col])

            #df['date_col'] = pd.to_datetime(df.index)
            #df['date_col'] = pd.to_datetime(df['date_col'])
            #df['ds'] = df['datetime_created'].dt.to_period('M')

            _="""
            freq = st.selectbox('Select date interval',('D','W','M','Y'))
            if freq == 'D':
                df['datetime_created'] = df['datetime_created'].dt.to_period('D')
    
            if freq == 'W':
                df['datetime_created'] = df['datetime_created'].dt.to_period('W')
    
            if freq == 'M':
                df['datetime_created'] = df['datetime_created'].dt.to_period('M')
    
            if freq == 'Y':
                df['datetime_created'] = df['datetime_created'].dt.to_period('Y')
            """


        else:
            date_col = st.selectbox("Select a column that contains the date-variable or similar", df.columns)
            #Convert the date column to datetime format

            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=[date_col], ascending=False)
            except:
                st.warning("Select a datetime variable")
                st.stop()

        st.write("")

        st.write("")
        st.write(""" ### Select a target variable to forecast""")
        # Create selectbox with numerical columns
        if numerical_cols:
            target_col = st.selectbox("Select the column that contains the target value", numerical_cols)
            target_Name = str(target_col) #Keep a column with original naming
            #st.write(target_Name)
        else:
            st.write("No numerical columns available in the DataFrame.")



        #target_col = st.selectbox("Select the column that contains the target value", df.columns)

        st.write("")

        filterVariable = ""
        filterVariable2 = ""

        st.write("")
        st.divider()

        # Choose a Filtervariable if useful #######
        optionCol1, optionCol2 = st.columns(2)

        if object_cols:
            object_filter = optionCol1.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols)
            if object_filter !=[]:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionCol2.multiselect("Choose the value(s) to keep", object_filter_selection)
                if object_selection !=[]:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable = "   >> Filter: " + str(object_selection)


        optionCol3, optionCol4 = st.columns(2)

        if object_cols:
            object_filter = optionCol3.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols, key="optioncol3")
            if object_filter !=[]:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionCol4.multiselect("Choose the value(s) to keep", object_filter_selection, key="optioncol3b")
                if object_selection !=[]:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable2 = " >> " + str(object_selection)



        st.divider()

        df['Target_original'] = df[target_col] #Keep a column with original naming

        # Rename the columns to ds and y as required by prophet
        df['Target'] = df[target_col]


        if createDateTime:
            df = df.rename(columns={createDateTime: "ds", target_col: "y"})
        else:
            df = df.rename(columns={date_col: "ds", target_col: "y"})
            df.sort_values(by=['ds'], ascending=True, inplace=True)

        if st.checkbox("Add rows where dates are missing"):
            # Get the minimum and maximum dates in the 'ds' column
            min_date = df['ds'].min()
            max_date = df['ds'].max()

            # Generate a range of dates for the months in-between
            new_dates = pd.date_range(start=min_date, end=max_date, freq='MS') + pd.DateOffset(
                days=1)  # Adding 1 day to get the start of next month

            # Create a new DataFrame with 'ds' column containing new_dates
            new_rows = pd.DataFrame({'ds': new_dates})

            # Append the new rows to the original DataFrame
            df = pd.concat([df, new_rows]).sort_values('ds').reset_index(drop=True)





        if st.checkbox("Interpolate missing values using linear interpolation"):
            df['y'] = df['y'].interpolate(method='linear')
            df['Target'] = df['Target'].interpolate(method='linear')

        st.divider()


        st.write("")
        historicalTableExpander = st.expander("Table with historical data")
        with historicalTableExpander:
            df_history = df.copy()

            #df_history = df_history.set_index('date_col')

            #df_history = df_history.sort_index(ascending=False)


            #df_history.sort_values(by=['ds'], ascending=False, inplace=True)
            st.write("df_history", df_history)

        startHistoricData = df['ds'].min()
        endHistoricData = df['ds'].max()
        #measurementsHistoricData = len(df['y'])

        #testAnzahl = pd.DataFrame()
        #testAnzahl['AccountNum'] = df['Target'].astype(float)

        #measurementsHistoricData = df.count(['y'])
        #measurementsHistoricData = len(testAnzahl)
        measurementsHistoricData= df['Target'].notna().sum()

        historycol1, historycol2, historycol3 = st.columns(3)
        with historycol1:
            st.write("Measurements: ",measurementsHistoricData)
        with historycol2:
            st.write("From: ", startHistoricData)
        with historycol3:
            st.write("To: ", endHistoricData)

        st.write("")
        st.divider()

        # Let the user specify the number of periods and the frequency for the forecast
        periods = st.number_input("Enter the number of periods for the forecast", min_value=1, value=10)
        freq = st.selectbox("Select the frequency of the forecast", ["D", "W", "M", "Q", "Y"],index=2,
       placeholder="Day, Week Month,..")



        st.divider()
        st.write("")

        st.subheader("")
        if st.button("Start prophet Forecasting"):

            # Create and fit a prophet model with default parameters
            model = ph.Prophet()
            model.fit(df)

            # Make a future dataframe with the specified periods and frequency
            future = model.make_future_dataframe(periods=periods, freq=freq)

            # Make a forecast with the model
            Prohetforecast_df = model.predict(future)

            # Show the forecasted values
            st.subheader("")
            st.subheader("Forecasted values for the next {} periods:".format(periods))

            st.dataframe(Prohetforecast_df[["ds", "yhat", "yhat_lower", "yhat_upper","trend"]].tail(periods))


            endValue_Yhat = Prohetforecast_df['yhat'].tail(1).round(1)
            endValue_trend = Prohetforecast_df['trend'].tail(1).round(1)

            endValueCol1, endValueCol2 = st.columns(2)
            endValueCol1.metric(label ="Target end value " + target_col, value=endValue_Yhat)

            endValueCol2.metric(label="Trend end value ", value=endValue_trend)


            #Prohetforecast_df = Prohetforecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            #st.dataframe(Prohetforecast_df)

            infoExpander = st.expander("Infos")
            with infoExpander:
                st.write("""
                
                In Facebook's Prophet forecasting model, these variables are related to the forecasted values and trends:
    
                yhat: This represents the forecasted value for the target variable. It's the estimated value of the variable at a specific point in time based on the historical data and the model's predictions.
                
                yhat_lower: This is the lower bound of the forecasted value. It represents the lower boundary of the prediction interval. In other words, it gives you an estimate of the lowest possible value the target variable could take at a specific point in time, based on the forecast.
                
                yhat_upper: This is the upper bound of the forecasted value. It represents the upper boundary of the prediction interval. Similar to yhat_lower, it gives you an estimate of the highest possible value the target variable could take at a specific point in time, based on the forecast.
                
                trend: This variable represents the underlying trend in the data. Prophet decomposes time series data into trend, seasonality, and holidays. The trend component captures the long-term direction or pattern in the data, which could be increasing, decreasing, or stable over time. It's essentially the long-term movement or tendency observed in the target variable.
                
                These variables together provide insights into the forecasted values, their uncertainty (represented by the prediction intervals), and the underlying trend in the time series data.
                                
                """)


            if len(Prohetforecast_df) > 0:
                def to_excel(Prohetforecast_df):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    Prohetforecast_df.to_excel(writer, index=True, sheet_name='Sheet1')
                    workbook = writer.book
                    worksheet = writer.sheets['Sheet1']
                    format1 = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column('A:A', None, format1)
                    writer.close()
                    processed_data = output.getvalue()
                    return processed_data


                df_xlsx = to_excel(Prohetforecast_df)
                st.download_button(label='📥 Save table with forecasted data to Excel?',
                                   data=df_xlsx,
                                   file_name= target_col + ' - forecasted' + '.xlsx')



            # Plot the forecast
            st.subheader("Plot of the forecast:")
            st.info("Target: " + target_col + filterVariable + filterVariable2)

            fig = model.plot(Prohetforecast_df)
            st.pyplot(fig)

            # Plot the forecast components
            st.subheader("Plot of the forecast components:")
            fig2 = model.plot_components(Prohetforecast_df)
            st.pyplot(fig2)

            #MERGER Forecast and orginal Data  #######################

            #clean up Prohetforecast_df
            df_prophetForecast = Prohetforecast_df[["ds", "yhat", "yhat_lower", "yhat_upper","trend"]]


            # Merge the two DataFrames on the common column 'ds'

            #df_history['Kind'] = "Historical data"
            #df_prophetForecast['Kind'] = "Forecasted data"


            merged_df = pd.merge(df_history, df_prophetForecast, on='ds', how='outer')

            merged_df.sort_values(by=['ds'], ascending=True, inplace=True)

            st.write("")
            st.write("Table with all historical and forecasted data of the target(yhat)")
            st.write(merged_df)

            if len(merged_df) > 0:
                def to_excel(merged_df):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    merged_df.to_excel(writer, index=True, sheet_name='Sheet1')
                    workbook = writer.book
                    worksheet = writer.sheets['Sheet1']
                    format1 = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column('A:A', None, format1)
                    writer.close()
                    processed_data = output.getvalue()
                    return processed_data


                df_xlsx = to_excel(merged_df)
                st.download_button(label='📥 Save table to Excel?',
                                   data=df_xlsx,
                                   file_name= target_col + ' - Historic and forecasted' + '.xlsx')



            st.subheader("")

            st.subheader("Chart with historical and forecasted data (yhat)")
            st.info("Target: " + target_col + filterVariable + filterVariable2)

            lineShapeAuswahl = 'linear'
            #lineShapeAuswahl = 'spline'

            forecastYhatMax = merged_df['yhat'].max()
            forecastTargetMax = merged_df['Target'].max()

            Ymax = forecastTargetMax * 1.2
            if forecastYhatMax > forecastTargetMax:
                Ymax = forecastYhatMax * 1.2



            forecastYhatMin = merged_df['yhat'].min()
            forecastTargetMax = merged_df['Target'].min()

            #lineShapeAuswahl = 'spline'
            lineShapeAuswahl = 'linear'
            #if st.checkbox("Lineare Verläufe"):
            #    lineShapeAuswahl = 'linear'

            figPlotlyLineChart_prophetForcast = px.line(merged_df, x='ds',
                                                                           y=['yhat','Target'],
                                                                            #y=['yhat', 'Target', 'trend'],
                                                                           line_shape=lineShapeAuswahl,
                                                                           # color_discrete_map={'GESAMTReichweite' : FARBE_GESAMT,'TVReichweite' : FARBE_TV,'ZATTOOReichweite' : FARBE_ZATTOO,'KINOReichweite' : FARBE_KINO,'DOOHReichweite' : FARBE_DOOH,'OOHReichweite' : FARBE_OOH,'FACEBOOKReichweite' : FARBE_FACEBOOK,'YOUTUBEReichweite' : FARBE_YOUTUBE,'ONLINEVIDEOReichweite' : FARBE_ONLINEVIDEO,'ONLINEReichweite' : FARBE_ONLINE, 'RADIOReichweite' : FARBE_RADIO},
                                                                           markers=True,
                                                                           # Animation:
                                                                           # range_x=[0, gesamtBudget*1000],
                                                                           range_y=[0, Ymax],
                                                                           # animation_frame="ds)
                                                                           )

            # Change grid color and axis colors
            figPlotlyLineChart_prophetForcast.update_xaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')
            figPlotlyLineChart_prophetForcast.update_yaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')

            st.plotly_chart(figPlotlyLineChart_prophetForcast, use_container_width=True)


            ###

            st.subheader("Chart with historical and trend of forecasted data (trend)")
            st.info("Target: " + target_col + filterVariable)

            #lineShapeAuswahl = 'spline'
            lineShapeAuswahl = 'linear'
            #if st.checkbox("Lineare Verläufe"):
            #    lineShapeAuswahl = 'linear'

            figPlotlyLineChart_prophetForcast2 = px.line(merged_df, x='ds',
                                                                           y=['trend','Target'],
                                                                            #y=['yhat', 'Target', 'trend'],
                                                                           line_shape=lineShapeAuswahl,
                                                                           # color_discrete_map={'GESAMTReichweite' : FARBE_GESAMT,'TVReichweite' : FARBE_TV,'ZATTOOReichweite' : FARBE_ZATTOO,'KINOReichweite' : FARBE_KINO,'DOOHReichweite' : FARBE_DOOH,'OOHReichweite' : FARBE_OOH,'FACEBOOKReichweite' : FARBE_FACEBOOK,'YOUTUBEReichweite' : FARBE_YOUTUBE,'ONLINEVIDEOReichweite' : FARBE_ONLINEVIDEO,'ONLINEReichweite' : FARBE_ONLINE, 'RADIOReichweite' : FARBE_RADIO},
                                                                           markers=True,
                                                                           # Animation:
                                                                           # range_x=[0, gesamtBudget*1000],
                                                                           range_y=[0, Ymax],
                                                                           # animation_frame="ds)
                                                                           )

            # Change grid color and axis colors
            figPlotlyLineChart_prophetForcast2.update_xaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')
            figPlotlyLineChart_prophetForcast2.update_yaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')

            st.plotly_chart(figPlotlyLineChart_prophetForcast2, use_container_width=True)









            comparisonCol1, comparisonCol2, comparisonCol3, comparisonCol4  = st.columns(4)
            with comparisonCol1:
                st.info("Stats of historical data and ALL modeled data")
                testcolumns = ['ds','yhat','Target']
                df_test = merged_df[testcolumns]
                #st.write(df_test)
                st.write(df_test.describe())


            with comparisonCol2:
                st.info("Stats of historical data and modeled historical data")
                # Filter rows where 'Target' column is not null
                df_test_historical = df_test[df_test['Target'].notnull()]

                # Reset index if needed
                df_test_historical.reset_index(drop=True, inplace=True)

                #st.write(df_test_historical)
                st.write(df_test_historical.describe())

            with comparisonCol3:
                st.info("Stats of forecasted data")

                #foreCast_selectedData = df_prophetForecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
                foreCast_selectedDataYhat = df_prophetForecast[["yhat"]].tail(periods)
                #st.write(foreCast_selectedData)
                st.write(foreCast_selectedDataYhat.describe())


            with comparisonCol4:
                st.info("Stats of historical data")

                testcolumns_original = ['ds','Target_original']
                df_test_historical_original = merged_df[testcolumns_original]

                #df_test_historical_original = df_test_historical_original[df_test_historical_original['Target_original'].notnull()]
                  # Reset index if needed
                df_test_historical_original.reset_index(drop=True, inplace=True)

                #st.write(df_test_historical)
                st.write(df_test_historical_original.describe())              



if option =="Statsmodels":
    st.title("Forecasting with statsmodels")

    statsModelsInfo = st.expander("Infos about statsmodels")
    with statsModelsInfo:
        st.markdown(""" 
        			   Die Python-Bibliothek statsmodels ist eine leistungsstarke Sammlung von Werkzeugen zur Schätzung von statistischen Modellen, Durchführung von statistischen Tests und Exploration von Daten. Sie bietet eine breite Palette von Funktionen für die Durchführung verschiedener statistischer Analysen, einschließlich lineare Regression, Zeitreihenanalyse, generalisierte lineare Modelle, robuste lineare Modelle, nichtlineare Modelle und vieles mehr.

        				Einige der wichtigsten Funktionen und Methoden von statsmodels sind:

        				Lineare Regression: Mit statsmodels können Sie lineare Regressionsmodelle erstellen und schätzen. Dies umfasst einfache lineare Regression, multiple lineare Regression und robuste lineare Regression.

        				Zeitreihenanalyse: Die Bibliothek bietet Funktionen zur Analyse von Zeitreihendaten, einschließlich ARIMA-Modelle (Autoregressive Integrated Moving Average), SARIMA-Modelle (Seasonal Autoregressive Integrated Moving Average) und vieles mehr.

        				Generalisierte lineare Modelle (GLM): statsmodels unterstützt die Schätzung von GLMs, die eine Erweiterung der linearen Regression sind und Modelle für Daten mit nicht-normalverteilten Fehlertermen ermöglichen.

        				Nichtparametrische Tests: Sie bietet auch eine Vielzahl von nichtparametrischen Tests wie den Kruskal-Wallis-Test, den Mann-Whitney-U-Test und den Kolmogorov-Smirnov-Test.

        				Explorative Datenanalyse (EDA): Durch Visualisierungen und Zusammenfassungen können Sie schnell Einblicke in Ihre Daten gewinnen und wichtige Muster oder Ausreißer identifizieren.

        				Multivariate Statistik: statsmodels ermöglicht die Durchführung von multivariaten statistischen Analysen wie Hauptkomponentenanalyse (PCA) und Faktorenanalyse.

        				Modellevaluation und Diagnostik: Die Bibliothek bietet Funktionen zur Bewertung von Modellen und zur Diagnose von Problemen wie Multikollinearität, Autokorrelation und Heteroskedastizität.""")


    n = st.slider('Number of Periods/month to forcast', min_value=1, max_value=24, value=12, step=1)

    # Filter numerical columns
    numerical_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    object_cols = df.select_dtypes(include='object').columns.tolist()

    st.write("The following numeric variables will be predicted: ",numerical_cols)



    date_col = st.selectbox("Select a column that contains the date-variable or similar", df.columns, key="statsmodelsagain")
    # Convert the date column to datetime format

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=[date_col], ascending=False)
    except:
        st.warning("Select a datetime variable")
        st.stop()






    st.divider()

    # Choose a Filtervariable if useful #######
    optionCola, optionColb = st.columns(2)



    if 1==1:
        if object_cols:
            object_filter = optionCola.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols)
            if object_filter != []:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionColb.multiselect("Choose the value(s) to keep", object_filter_selection)
                if object_selection != []:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable3 = "   >> Filter: " + str(object_selection)


    optionColC, optionColD= st.columns(2)

    if 1==1:
        if object_cols:
            object_filter = optionColC.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols, key="optionColC")
            if object_filter != []:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionColD.multiselect("Choose the value(s) to keep", object_filter_selection, key="optionColD")
                if object_selection != []:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable4 = " >> " + str(object_selection)



    st.divider()


    st.subheader("")

    if st.button("Start statsmodes forecast"):

        df = df.sort_values(by=[date_col], ascending=True)

        forecast_df = pd.DataFrame()

        # Specify the order of the ARIMA model (p, d, q)
        p, d, q = 1, 1, 1

        for column in df[numerical_cols]:
            model = ARIMA(df[column], order=(p, d, q))
            model_fit = model.fit()
            forecasted_values = model_fit.forecast(steps=n)  # 'n' is the number of months to forecast

            # Create a new DataFrame for the forecasted values for the current column
            forecast_df[column] = forecasted_values

        forecast_df["Month Year"] = pd.date_range(
                start=df[date_col].max() + pd.DateOffset(months=1), periods=n, freq='M')

        st.subheader("")

        st.subheader("Measured and forecasted Values by statsmodels:")

        StatsModelsForecast_df = df.append(forecast_df)



        StatsModelsForecast_df[date_col] = StatsModelsForecast_df[date_col].dt.strftime('%Y-%m')
        StatsModelsForecast_df['Month Year'] = StatsModelsForecast_df['Month Year'].dt.strftime('%Y-%m')


        # Fill missing values in 'date_col' with values from 'Month Year', and vice versa
        StatsModelsForecast_df['Time'] = StatsModelsForecast_df[date_col].fillna(StatsModelsForecast_df['Month Year'])
        StatsModelsForecast_df['Time'] = StatsModelsForecast_df['Time'].fillna(StatsModelsForecast_df[date_col])

        #StatsModelsForecast_df['Time'] = StatsModelsForecast_df[date_col] + StatsModelsForecast_df["Month Year"]

        st.subheader("")

        st.write(StatsModelsForecast_df)

        if len(StatsModelsForecast_df) > 0:
            def to_excel(StatsModelsForecast_df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                StatsModelsForecast_df.to_excel(writer, index=True, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.close()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(StatsModelsForecast_df)
            st.download_button(label='📥 Save table with statsmodels values to Excel?',
                               data=df_xlsx,
                               file_name=' Statsmodels values - Historic and forecasted' + '.xlsx')
            

        st.write("")    
        showforecastVariable = st.selectbox('Show selected forecasted value', numerical_cols)




        if len(StatsModelsForecast_df) > 0:
            figPlotlyLineChart_statsmodelsForcast = px.line(StatsModelsForecast_df, x='Time',
                                                                           y=[showforecastVariable],
                                                                            #y=['yhat', 'Target', 'trend'],
                                                                           #line_shape=lineShapeAuswahl,
                                                                           # color_discrete_map={'GESAMTReichweite' : FARBE_GESAMT,'TVReichweite' : FARBE_TV,'ZATTOOReichweite' : FARBE_ZATTOO,'KINOReichweite' : FARBE_KINO,'DOOHReichweite' : FARBE_DOOH,'OOHReichweite' : FARBE_OOH,'FACEBOOKReichweite' : FARBE_FACEBOOK,'YOUTUBEReichweite' : FARBE_YOUTUBE,'ONLINEVIDEOReichweite' : FARBE_ONLINEVIDEO,'ONLINEReichweite' : FARBE_ONLINE, 'RADIOReichweite' : FARBE_RADIO},
                                                                           markers=True,
                                                                           # Animation:
                                                                           # range_x=[0, gesamtBudget*1000],
                                                                           #range_y=[0, Ymax],
                                                                           # animation_frame="ds)
                                                                           )

            # Change grid color and axis colors
            figPlotlyLineChart_statsmodelsForcast.update_xaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')
            figPlotlyLineChart_statsmodelsForcast.update_yaxes(showline=True, linewidth=0.1,
                                                                              linecolor='Black', gridcolor='Black')

            st.plotly_chart(figPlotlyLineChart_statsmodelsForcast, use_container_width=True)






if option =="Trendchart":
    st.title("Scatter Trend Chart")

    trendChartInfo = st.expander("Infos")
    with trendChartInfo:
        st.markdown(""" The chart can show the following trend lines:



OLS (Ordinary Least Squares): OLS is a method for estimating the unknown parameters in a linear regression model.
In the context of a scatter plot, the OLS trend line represents the "best-fit" straight line through the data points.
This line minimizes the sum of the squared vertical distances (residuals) between each data point and the line.
OLS assumes a linear relationship between the variables and may not capture non-linear trends well.
LOWESS (Locally Weighted Scatterplot Smoothing):

LOWESS is a non-parametric method used to fit a smooth curve through a scatter plot.
Unlike OLS, LOWESS does not assume a specific functional form for the relationship between the variables.
Instead, it fits a series of local polynomial regressions to subsets of the data.
The resulting trend line is smoother and can capture non-linear relationships in the data better than OLS.
The level of smoothing can be adjusted using a bandwidth parameter.
Expanding Mean:

Expanding Mean is a simple method of calculating a trend line by taking the mean of all the data points up to a certain point.
As the name suggests, it involves continuously expanding the window of data points used to calculate the mean.
This trend line provides a simple representation of the overall trend in the data over time.
However, it may not capture short-term fluctuations or changes in the trend direction effectively.
                    
Each of these trend lines offers a different approach to summarizing the relationship between variables in a scatter plot. The choice of which one to use may depend on the characteristics of your data and the specific insights you're trying to extract from the visualization.

OLS, LOWESS, and Expanding Mean) are commonly used for fitting trends to scatter plots, especially in statistical analysis and data visualization. 

Further trendlines - not (yet) implemented here:

However, there are other types of trendlines that could be considered depending on the specific requirements of your application and the characteristics of your data. Here are a few additional options:

 Rolling Mean (rolling):The rolling mean is a method of smoothing a time series data by calculating the mean of consecutive subsets of data points over a specified window (rolling window).
This method helps to reduce the effects of short-term fluctuations or noise in the data, revealing underlying trends or patterns.
The rolling window size determines the number of data points included in each calculation of the mean, and it can be adjusted based on the frequency and characteristics of the data.
Rolling mean is useful for identifying longer-term trends while preserving the overall shape of the data.
                    
Exponentially Weighted Moving Average (ewm): The exponentially weighted moving average (EWMA) is another method of smoothing a time series data, similar to the rolling mean but with an exponential weighting scheme.
In EWMA, recent data points are given more weight compared to older data points, with the weights decreasing exponentially as you move further back in time.
This weighting scheme allows EWMA to react more quickly to changes in the data compared to simple rolling mean, making it suitable for capturing short-term trends or detecting abrupt changes.
Like the rolling mean, the smoothing parameter (often referred to as the smoothing factor or span) in EWMA determines the rate at which older observations decay in influence.
In summary, "rolling" and "ewm" are both methods of smoothing time series data to reveal underlying trends or patterns, with "rolling" using a fixed-size window for averaging and "ewm" giving more weight to recent observations. They are useful tools for data preprocessing and visualization in time series analysis.                   


Exponential Trendline:Fits an exponential curve to the data points.
Useful for data that exhibits exponential growth or decay.

Power Trendline:Fits a power law curve (y = a * x^b) to the data points.
Useful for data where the relationship between the variables follows a power law.

Logarithmic Trendline:Fits a logarithmic curve (y = a * ln(x) + b) to the data points.
Useful for data that shows diminishing returns or exponential decay.

Polynomial Trendline:Fits a polynomial curve (e.g., quadratic, cubic) to the data points.
Useful for capturing non-linear relationships with higher order terms.

Moving Average Trendline: Computes the moving average of the data points over a specified window.
Useful for smoothing out short-term fluctuations and identifying longer-term trends.                    
                          
                    
                    """)


    #n = st.slider('Number of Periods/month to forcast', min_value=1, max_value=24, value=12, step=1)

    # Filter numerical columns
    numerical_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    object_cols = df.select_dtypes(include='object').columns.tolist()





    date_col = st.selectbox("Select a column that contains the date-variable or similar", df.columns, key="statsmodelsagain")
    # Convert the date column to datetime format

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=[date_col], ascending=False)
    except:
        st.warning("Select a datetime variable")
        st.stop()

    if len(df)>5:
        st.write("")

        st.write("")
        st.write(""" ### Select a target variable""")
        # Create selectbox with numerical columns
        if numerical_cols:
            target_col = st.selectbox("Select the column that contains the target value", numerical_cols)
            target_Name = str(target_col) #Keep a column with original naming
            #st.write(target_Name)
        else:
            st.write("No numerical columns available in the DataFrame.")



    st.divider()

    # Choose a Filtervariable if useful #######
    optionCola, optionColb = st.columns(2)



    if 1==1:
        if object_cols:
            object_filter = optionCola.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols)
            if object_filter != []:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionColb.multiselect("Choose the value(s) to keep", object_filter_selection)
                if object_selection != []:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable3 = "   >> Filter: " + str(object_selection)


    optionColC, optionColD= st.columns(2)

    if 1==1:
        if object_cols:
            object_filter = optionColC.selectbox("Optional - Choose to filter a (non numerical) variable", object_cols, key="optionColC")
            if object_filter != []:
                object_filter_selection = df[object_filter].unique()
                object_selection = optionColD.multiselect("Choose the value(s) to keep", object_filter_selection, key="optionColD")
                if object_selection != []:
                    df_filtered = df[df[object_filter].isin(object_selection)]
                    df = df_filtered
                    filterVariable4 = " >> " + str(object_selection)



    st.divider()


    st.subheader("")

    if st.checkbox("Show Trend Chart"):

        df = df.sort_values(by=[date_col], ascending=True)

        #st.write("df:", df)

        if 1==1:
            st.subheader("Trend for " + target_Name)

            trendline_options = ["ols", "lowess", "expanding"]
            trendline_names = {
                "ols": "Ordinary Least Squares (OLS)",
                "lowess": "Locally Weighted Scatterplot Smoothing (LOWESS)",
                "expanding": "Expanding Mean",
                #"rolling": "Rolling Mean",
                #"ewm": "Exponentially Weighted Moving Average"
                }

            trendlineChoice = st.selectbox("Choose Trendline", trendline_options)

            st.markdown(trendline_names.get(trendlineChoice, "Unknown Trendline"))

            YRangeCol1, YRangeCol2, YRangeCol3 = st.columns([0.1, 0.1, 0.8])

            yMinRange = df[target_col].min()*0.5
            yMaxRange = df[target_col].max() * 1.5

            with YRangeCol1:
                yMinInput = st.number_input("YMin", placeholder="Ymin", value=yMinRange)
            with YRangeCol2:
                yMaxInput = st.number_input("YMax", placeholder="Ymax", value=yMaxRange)

            with YRangeCol3:
                st.write("")

            figPlotlyScatterchartwithTrendline = px.scatter(df,
                                                         x=date_col,
                                                         y=target_col,
                                                         trendline = trendlineChoice,
                                                         range_y=[yMinInput, yMaxInput],
                                                         )


            st.plotly_chart(figPlotlyScatterchartwithTrendline, use_container_width=True)


            # Extracting x, y, and trendline values
            x_values = figPlotlyScatterchartwithTrendline.data[0].x
            y_values = figPlotlyScatterchartwithTrendline.data[0].y
            trendline_values = (figPlotlyScatterchartwithTrendline.data[1].y)
            #st.write(trendline_values)
            #st.write(len(trendline_values))



            # Create dataframe
            Trend_data = {'x': x_values, 'y': y_values,'trend': trendline_values}
            df_trendline = pd.DataFrame(Trend_data)

            st.info("Table including the data from the trend line")
            # Print or further process df_trendline
            st.dataframe(df_trendline)
            st.write("Cases: ", len(df_trendline))

            if len(df_trendline) > 0:
                def to_excel(df_trendline):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    df_trendline.to_excel(writer, index=True, sheet_name='Sheet1')
                    workbook = writer.book
                    worksheet = writer.sheets['Sheet1']
                    format1 = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column('A:A', None, format1)
                    writer.close()
                    processed_data = output.getvalue()
                    return processed_data


                df_xlsx = to_excel(df_trendline)
                st.download_button(label='📥 Save table to Excel?',
                                   data=df_xlsx,
                                   file_name= target_col + ' - Historic Trend - ' + trendlineChoice +'.xlsx')
                
                _="""
                if st.checkbox("Forecast with linear regression"):
                    #forecasting von ols mit linear regression

                    # Group by X column and calculate mean of Y column for each group
                    df_trendline = df_trendline.groupby('x')['trend'].mean().reset_index()

                    forecast_period = st.slider('Number of Periods/month to forcast', min_value=1, max_value=24, value=12, step=1)

                    X = df_trendline['x'].values.reshape(-1, 1)  # Features (date)
                    y = df_trendline['trend'].values  # Target variable

                    model = LinearRegression()
                    model.fit(X, y)

                    # Forecast future values
                    last_date = df[date_col].max()
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period + 1, closed='right')
                    forecast_dates = forecast_dates[1:]  # Exclude last_date since it's already in the data
                    forecast_X = np.array(range(len(df) + 1, len(df) + forecast_period + 1)).reshape(-1, 1)  # Future dates

                    forecast_values = model.predict(forecast_X)

                    # Create dataframe for forecasted values
                    forecast_df = pd.DataFrame({
                        date_col: forecast_dates,
                        target_col: forecast_values
                    })

                    # Display forecasted values
                    st.write("Forecasted Values:")
                    st.write(forecast_df)
                    """