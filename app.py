import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import squarify
import pickle
from datetime import datetime

st.set_page_config(page_title='Customer Segmentation Program', layout='wide')

st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    .css-1aumxhk {
        background-color: #ffe0e0;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Setting the title of the app
st.title("Project 3: Customer Segmentation Analysis")

custom_css = """
    <style>
        .css-5sror9 {
            font-family: 'Roboto';
            font-weight: normal !important;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Create the sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Project Introduction", 'Project Insight', "New RFM Analysis"]
                           , icons=['house', 'book', 'upload']
                           , menu_icon="cast"
                           , default_index=0
                           ,)

st.subheader(f":blue[{selected}]")


# Load pre-trained scaler and KMeans model
scaler = pickle.load(open('scaler.pkl', 'rb'))
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))

# Map  clusters 
cluster_labels = {0: 'Regular Customer', 1: 'Lost Customer', 2: 'Need Attention Customer', 3: 'Star Customer'}

def plot_treemap(cluster_summary):
    # Define colors for each segment
    colors_dict = {
        'Regular Customer': 'green',
        'Lost Customer': 'yellow',
        'Need Attention Customer': 'cyan',
        'Star Customer': 'red'
    }
    colors = [colors_dict[segment] for segment in cluster_summary.index]

    # Generate labels for each segment in the treemap
    labels = [
        f'{name}\n{row["Recency"]:,.0f} days\n{row["Frequency"]:,.0f} orders\n${row["Monetary"]:,.2f}\n{row["Count"]:.0f} customers ({row["Percentage"]:.2f}%)'
        for name, row in cluster_summary.iterrows()
    ]

    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(12, 8)

    squarify.plot(
        sizes=cluster_summary['Count'], 
        color=colors, 
        label=labels,
        alpha=0.5, 
        text_kwargs={'fontsize':11, 'weight':'bold'}
    )
    plt.title('Customer Segments', fontsize=20, fontweight='bold')
    plt.axis('off')
    st.pyplot(fig)

if selected == "Project Introduction":
    bullet_list_html = """
        <h2><p><span style="text-decoration: underline; color: SeaGreen;"> <strong> Business Objective:&nbsp; </strong> </span></p></h2>
        RFM analysis is a marketing technique to help businesses categorize customers into segments, enable targeted and personalized marketing strategies. 
        <p>
        <h2><p><span style="text-decoration: underline; color: SeaGreen;"> <strong> Method:&nbsp; </strong> </span></p></h2>
        Analyzing customer behavior based on three key factors:
        <ul style="list-style-type: circle;">
            <li><strong> Recency (R): </strong> How recent was the customer's last purchase?</li>
            <li><strong> Frequency (F): </strong> How often a customer has transacted or interacted with the brand during a particular period of time? <br /> </li>
            <li><strong> Monetary (M): </strong> How much a customer has spent with the brand during a particular period of time? </li>
        </ul>
        """
    st.image("images/rfm pur.png", width=700)
    st.markdown(bullet_list_html, unsafe_allow_html=True)
    st.image("images/rfm_cover.png", width=800)

##################################
if selected == "Project Insight":
    st.write("#### :blue[1. Dataset Exploration:]")
    st.write("Dataset sample:")
    df_t = pd.read_csv('OnlineRetail_t.csv')
    st.dataframe(df_t.head(7))
    shape="""Dataset includes <span style="color: #339966;"><strong>541,909</strong> </span>rows and <span style="color: #339966;">8</span> columns
             <p>
             """
    st.markdown(shape, unsafe_allow_html=True)
    st.write("##                               ")
    st.write("##### * Explore numeric columns:")
    st.image("images/quantity.png", width=1300)  
    st.write("##                              ")
    st.image("images/unit price.png", width=1300)  
    st.write("##                              ")
    st.write("##### * Explore categorical columns:")
    st.image("images/country.png", width=800)  
    st.image("images/month.png", width=900)  
    st.write("##### * Comment:")
    st.write("Quantity: dataset có những single transaction mà outlier quantity lớn như Quantity =4300, 738, 552, 906, 80995\n còn lại chiếm đa số các giao dịch số lượng Quantity nhỏ lẻ= 1,12,2,6,4,3... \n")
    st.write("Unitprice:Có những single transaction mà outlier Unitprice cao như Unitprice=8142, 3949, 4161")
    
    st.write("##                              ")
    st.write("#### :blue[2. Model Customer Segmentation:]")
    st.write("##### * RFM Distribution")
    st.image("images/skew.png", width=800)  
    st.write("##### * Perform: Elbow method, K-means machine learning for data clustering")
    st.write("##### * Cluster Output")
    df_o = pd.read_csv('out.csv')
    df_o = df_o.drop(columns=['Unnamed: 0'])
    st.dataframe(df_o)
    st.image("images/Treemap.png", width=800)  
    st.image("images/scatter plot.png", width=1000)  
    st.write("##                              ")
    st.write("##### * Comment:")
    st.write("Cluster 0 (green)  -> Regular Customer: Khách hàng trong cụm này có R vừa phải (86 ngày) và F thấp (3 đơn hàng), với M vừa phải ($1.323) có thể mua sắm khá đều nhưng không chi tiêu vừa phải không nhiều, có thể là khách hàng thường xuyên, những người đóng góp liên tục theo thời gian. Nhóm này có thể được nhắm mục tiêu bằng các chiến lược tương tác nhằm tăng tần suất mua hàng của họ")
    st.write("Cluster 1 (yellow) -> Lost Customer: nhóm có R cao (287 ngày), F rất thấp (1 đơn hàng) và M thấp ($549). Đây có thể là những người mua một lần hoặc những khách hàng gần đây không tương tác và có khả năng trở thành khách hàng bị mất. Các chiến dịch tương tác lại và ưu đãi đặc biệt có thể được sử dụng để khuyến khích họ mua hàng nhiều hơn")
    st.write("Cluster 2 (cyan)   -> Need Attention Customer: Với R tương đối cao (177 ngày), F (2 đơn hàng) và M thấp ($653), phân khúc này tương tự như Cluster 1 nhưng tốt hơn một chút về F và M. Những khách hàng này gần đây không tương tác, tần suất và giá trị tiền tệ của họ không cao, có thể đang dần rời bỏ và có thể là mục tiêu của các nỗ lực giữ chân để ngăn họ rơi vào tình trạng không hoạt động")
    st.write("Cluster 3 (red)    -> Star Customer: Cụm này có R thấp nhất (22 ngày), F cao nhất (6 đơn hàng) và M cao nhất ($3,199). Đây là những khách hàng tốt nhất và trung thành nhất của bạn, những người mua sắm thường xuyên và chi tiêu nhiều nhất. Họ có thể được ưu tiên cho các chương trình VIP, phần thưởng dành cho khách hàng thân thiết và các ưu đãi độc quyền để duy trì mức độ tương tác cao")

#######################################
if selected == "New RFM Analysis":
    method = st.radio("###### Select method to input data:", ["RFM By Selection", "RFM By Upload File"])
    st.write("##                              ")
    if method == "RFM By Selection":
        #recency_1 = st.slider("Recency (days since last purchase)", 0, 365, 30)
        max_date = datetime.today().date()  # or a specific date from your data
        last_purchase_date = st.date_input("Select the last purchase date", max_date)
        recency= (max_date - last_purchase_date).days
        frequency = st.slider("Select total number of purchases (Frequency)", 1, 1000, 5)
        monetary = st.slider("Select total amount have spent (Monetary)", 0, 10000, 100)

        st.markdown(f"""
            <p><strong>There has been <span style='color: SeaGreen;'>{recency}</span> days since last purchase.</strong></p>
            <p><strong>Customer has bought <span style='color: SeaGreen;'>{frequency}</span> purchases so far.</strong></p>
            <p><strong>The total amount Customer has spent <span style='color: SeaGreen;'>{monetary}</span></strong></p>
            """, unsafe_allow_html=True)

        if st.button('Predict Customer Segment'):
            input_data = pd.DataFrame([[recency, frequency, monetary]], 
                                    columns=['Recency', 'Frequency', 'Monetary'])

            input_scaled = scaler.transform(input_data)
            cluster = kmeans_model.predict(input_scaled)
            input_data['Cluster'] = cluster
            input_data['Segment'] = input_data['Cluster'].map(cluster_labels)
            display_data = input_data.drop(columns=['Cluster'])
            st.write(display_data)        

    elif method == "RFM By Upload File":
        uploaded_file = st.file_uploader("Choose a CSV file for RFM segmentation", type="csv")
        
        if uploaded_file is not None:
            st.success('File uploaded successfully!')
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.write("Sample data have uploaded:")
            st.dataframe(df.head(7))
            df.drop_duplicates(inplace=True)
            df.dropna(subset=['CustomerID'], inplace=True)

            quantity_filter = df['Quantity'] > 0
            unit_price_filter = df['UnitPrice'] > 0
            df = df[quantity_filter & unit_price_filter]

            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df['SalesRevenue'] = df['UnitPrice'] * df['Quantity']

            max_date = df['InvoiceDate'].max().date()
            recency = lambda x: (max_date - x.max().date()).days
            frequency = lambda x: len(x.unique())
            monetary = lambda x: round(sum(x), 2)

            df_rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': recency,
                'InvoiceNo': frequency,
                'SalesRevenue': monetary
             }).reset_index()

            df_rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'SalesRevenue': 'Monetary'
            }, inplace=True)

            # Ensure only RFM features are used for scaling and prediction
            rfm_features = df_rfm[['Recency', 'Frequency', 'Monetary']]
            rfm_scaled = scaler.transform(rfm_features)

            # Predict the clusters and map to segments
            df_rfm['Cluster'] = kmeans_model.predict(rfm_scaled)
            df_rfm['Segment'] = df_rfm['Cluster'].apply(lambda x: cluster_labels[x])
            
            st.write("#### :blue[RFM Analysis Result:]")
            # first few rows of the segmented data
            st.write("###### Segmented RFM data (first 10 rows):")
            st.dataframe(df_rfm.head(10))

            # count of customers in each segment
            segment_counts = df_rfm['Segment'].value_counts()
            segment_percentage = round((segment_counts / segment_counts.sum() * 100),2).astype(str) + '%'
            segment_summary = pd.DataFrame({
                'Segment': segment_counts.index,
                'Count': segment_counts.values,
                'Percent': segment_percentage.values
            })

            segment_summary.set_index('Segment', inplace=True)
            st.write("###### Segmented Customer Distribution:")
            st.dataframe(segment_summary)
            # Calculate cluster summary for treemap visualization
            cluster_summary = df_rfm.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            })

            cluster_summary['Count'] = df_rfm['Segment'].value_counts()
            cluster_summary['Percentage'] = (cluster_summary['Count'] / df_rfm['Segment'].value_counts().sum()) * 100

            # Plot the treemap
            plot_treemap(cluster_summary)
