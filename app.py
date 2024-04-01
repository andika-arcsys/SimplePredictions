

#2024.04.01.18

import streamlit as st
import pandas as pd

#forecasting
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


st.set_page_config(
    page_title="Simple Preditictions",
    page_icon="🧊",
    #layout="wide",
    #initial_sidebar_state="expanded",
)



#---Option Menu -------------------

option = option_menu(
	menu_title="Simple Predictions",
	options=["Prophet", "Statsmodels"],
	icons=["0-square", "2-circle"], #https://icons.getbootstrap.com/
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
            st.info("Target: " + target_col + filterVariable)

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
            st.info("Target: " + target_col + filterVariable)

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









            comparisonCol1, comparisonCol2, comparisonCol3  = st.columns(3)
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


    #forecastVariable = st.selectbox('Value to forecast?', numerical_cols)

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
                    filterVariable = "   >> Filter: " + str(object_selection)




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

        st.subheader("Measured and forecasted Values by statsmodels:")

        StatsModelsForecast_df = df.append(forecast_df)

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