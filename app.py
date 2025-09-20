"""
🚢 Streamlit Application for Exploratory Data Analysis of the Titanic Dataset
============================================================================

This application loads the clean Titanic dataset (without missing values)
and allows interactive exploratory analysis using Plotly Express.

Key features:
- Data loading with cache for performance optimization
- Tab interface to organize functionalities
- Interactive filters in sidebar
- Dynamic visualizations with Plotly Express
- Filtered data download
- Defensive programming to handle errors

Installation:
    pip install streamlit pandas numpy plotly seaborn scikit-learn

Execution:
    streamlit run app.py

The expected CSV by default is ./titanic_clean.csv 
(or upload your own CSV in the sidebar)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
        page_title="Titanic EDA",
        layout="wide",
        page_icon="🚢"
)

# Function to load data with cache
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
        """Loads the clean Titanic dataset with cache optimization"""
        try:
                if uploaded_file is not None:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ Custom file loaded: {uploaded_file.name}")
                elif file_path:
                        df = pd.read_csv(file_path)
                        st.success(f"✅ Default file loaded: {file_path}")
                else:
                        # Fallback: create synthetic dataset if file not found
                        st.warning("⚠️ File not found. Creating synthetic dataset for demonstration.")
                        np.random.seed(42)
                        n_samples = 891
                        df = pd.DataFrame({
                                'survived': np.random.choice([0, 1], n_samples),
                                'pclass': np.random.choice([1, 2, 3], n_samples),
                                'sex': np.random.choice(['male', 'female'], n_samples),
                                'age': np.random.normal(30, 15, n_samples).clip(0, 100),
                                'sibsp': np.random.choice([0, 1, 2, 3], n_samples),
                                'parch': np.random.choice([0, 1, 2], n_samples),
                                'fare': np.random.exponential(30, n_samples),
                                'embarked': np.random.choice(['C', 'Q', 'S'], n_samples)
                        })
                
                return df
        except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return pd.DataFrame()

# Function to create filters in sidebar
def create_filters(df):
        """Creates interactive filters in the sidebar"""
        st.sidebar.header("🔍 Data Filters")
        
        # Initialize filters
        filters = {}
        
        # Sex filter
        if 'sex' in df.columns:
                sex_options = ['All'] + sorted(df['sex'].unique().tolist())
                filters['sex'] = st.sidebar.selectbox("Sex:", sex_options)
        
        # Class filter
        class_col = 'pclass' if 'pclass' in df.columns else 'class'
        if class_col in df.columns:
                class_options = ['All'] + sorted(df[class_col].unique().tolist())
                filters['class'] = st.sidebar.selectbox("Class:", class_options, key='class_filter')
        
        # Embarked port filter
        if 'embarked' in df.columns:
                embarked_options = ['All'] + sorted(df['embarked'].dropna().unique().tolist())
                filters['embarked'] = st.sidebar.selectbox("Port of Embarkation:", embarked_options)
        
        # Survival filter
        if 'survived' in df.columns:
                survived_options = st.sidebar.multiselect(
                        "Survival:",
                        options=[0, 1],
                        default=[0, 1],
                        format_func=lambda x: 'Did Not Survive' if x == 0 else 'Survived'
                )
                filters['survived'] = survived_options
        
        # Age range filters
        if 'age' in df.columns:
                age_min, age_max = float(df['age'].min()), float(df['age'].max())
                filters['age_range'] = st.sidebar.slider(
                        "Age Range:",
                        min_value=age_min,
                        max_value=age_max,
                        value=(age_min, age_max),
                        step=1.0
                )
        
        # Fare range filters
        if 'fare' in df.columns:
                fare_min, fare_max = float(df['fare'].min()), float(df['fare'].max())
                filters['fare_range'] = st.sidebar.slider(
                        "Fare Range:",
                        min_value=fare_min,
                        max_value=fare_max,
                        value=(fare_min, fare_max),
                        step=1.0
                )
        
        return filters

# Function to apply filters
def apply_filters(df, filters):
        """Applies selected filters to the DataFrame"""
        df_filtered = df.copy()
        
        # Sex filter
        if filters.get('sex') and filters['sex'] != 'All':
                df_filtered = df_filtered[df_filtered['sex'] == filters['sex']]
        
        # Class filter
        class_col = 'pclass' if 'pclass' in df.columns else 'class'
        if filters.get('class') and filters['class'] != 'All' and class_col in df.columns:
                df_filtered = df_filtered[df_filtered[class_col] == filters['class']]
        
        # Embarked port filter
        if filters.get('embarked') and filters['embarked'] != 'All':
                df_filtered = df_filtered[df_filtered['embarked'] == filters['embarked']]
        
        # Survival filter
        if filters.get('survived') and 'survived' in df.columns:
                df_filtered = df_filtered[df_filtered['survived'].isin(filters['survived'])]
        
        # Age filter
        if filters.get('age_range') and 'age' in df.columns:
                age_min, age_max = filters['age_range']
                df_filtered = df_filtered[
                        (df_filtered['age'] >= age_min) & (df_filtered['age'] <= age_max)
                ]
        
        # Fare filter
        if filters.get('fare_range') and 'fare' in df.columns:
                fare_min, fare_max = filters['fare_range']
                df_filtered = df_filtered[
                        (df_filtered['fare'] >= fare_min) & (df_filtered['fare'] <= fare_max)
                ]
        
        return df_filtered

# Function to create EDA plots
def create_eda_plots(df):
        """Creates main plots for EDA"""
        plots = {}
        
        try:
                # 1. Age histogram
                if 'age' in df.columns:
                        plots['age_hist'] = px.histogram(
                                df, 
                                x='age', 
                                nbins=30,
                                title='Age Distribution',
                                labels={'age': 'Age', 'count': 'Frequency'}
                        )
                        plots['age_hist'].update_layout(showlegend=False)
        except Exception as e:
                st.error(f"Error creating age histogram: {e}")
        
        try:
                # 2. Survival countplot by sex
                if 'survived' in df.columns and 'sex' in df.columns:
                        survival_by_sex = df.groupby(['sex', 'survived']).size().reset_index(name='count')
                        plots['survival_by_sex'] = px.bar(
                                survival_by_sex,
                                x='sex',
                                y='count',
                                color='survived',
                                title='Survival by Sex',
                                labels={'sex': 'Sex', 'count': 'Number of Passengers'},
                                color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
                        )
        except Exception as e:
                st.error(f"Error creating survival by sex plot: {e}")
        
        try:
                # 3. Survival countplot by class
                class_col = 'pclass' if 'pclass' in df.columns else 'class'
                if 'survived' in df.columns and class_col in df.columns:
                        survival_by_class = df.groupby([class_col, 'survived']).size().reset_index(name='count')
                        plots['survival_by_class'] = px.bar(
                                survival_by_class,
                                x=class_col,
                                y='count',
                                color='survived',
                                title='Survival by Class',
                                labels={class_col: 'Class', 'count': 'Number of Passengers'},
                                color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
                        )
        except Exception as e:
                st.error(f"Error creating survival by class plot: {e}")
        
        try:
                # 4. Fare boxplot by class
                class_col = 'pclass' if 'pclass' in df.columns else 'class'
                if 'fare' in df.columns and class_col in df.columns:
                        plots['fare_by_class'] = px.box(
                                df,
                                x=class_col,
                                y='fare',
                                title='Fare Distribution by Class',
                                labels={class_col: 'Class', 'fare': 'Fare'}
                        )
        except Exception as e:
                st.error(f"Error creating fare by class boxplot: {e}")
        
        try:
                # 5. Fare boxplot by port of embarkation
                if 'fare' in df.columns and 'embarked' in df.columns:
                        plots['fare_by_embarked'] = px.box(
                                df,
                                x='embarked',
                                y='fare',
                                title='Fare Distribution by Port of Embarkation',
                                labels={'embarked': 'Port of Embarkation', 'fare': 'Fare'}
                        )
        except Exception as e:
                st.error(f"Error creating fare by port boxplot: {e}")
        
        try:
                # 6. Correlation heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                        correlation_matrix = numeric_df.corr()
                        plots['correlation_heatmap'] = px.imshow(
                                correlation_matrix,
                                text_auto=True,
                                aspect='auto',
                                title='Correlation Matrix - Numeric Variables',
                                color_continuous_scale='RdBu_r'
                        )
        except Exception as e:
                st.error(f"Error creating correlation heatmap: {e}")
        
        return plots

# Function to create custom plots
def create_custom_plot(df, x_col, y_col, plot_type, hue_col=None):
        """Creates custom plots according to user selection"""
        try:
                if plot_type == 'Histogram':
                        if x_col in df.columns:
                                fig = px.histogram(df, x=x_col, color=hue_col, title=f'Histogram of {x_col}')
                                return fig
                
                elif plot_type == 'Bar (count)':
                        if x_col in df.columns:
                                if hue_col and hue_col in df.columns:
                                        count_data = df.groupby([x_col, hue_col]).size().reset_index(name='count')
                                        fig = px.bar(count_data, x=x_col, y='count', color=hue_col, 
                                                             title=f'Count of {x_col} by {hue_col}')
                                else:
                                        count_data = df[x_col].value_counts().reset_index()
                                        count_data.columns = [x_col, 'count']
                                        fig = px.bar(count_data, x=x_col, y='count', 
                                                             title=f'Count of {x_col}')
                                return fig
                
                elif plot_type == 'Boxplot':
                        if x_col in df.columns and y_col and y_col in df.columns:
                                fig = px.box(df, x=x_col, y=y_col, color=hue_col,
                                                     title=f'Boxplot of {y_col} by {x_col}')
                                return fig
                
                elif plot_type == 'Scatter':
                        if x_col in df.columns and y_col and y_col in df.columns:
                                fig = px.scatter(df, x=x_col, y=y_col, color=hue_col,
                                                             title=f'Scatter Plot: {x_col} vs {y_col}')
                                return fig
                
                return None
        
        except Exception as e:
                st.error(f"Error creating custom plot: {e}")
                return None

# MAIN APPLICATION
def main():
        """Main function of the Streamlit application"""
        
        # Main title
        st.title("🚢 Exploratory Data Analysis of the Titanic Dataset")
        st.markdown("---")
        st.markdown("**Clean Dataset** - No missing values after applying KNN imputation")
        
        # Sidebar for file upload
        st.sidebar.title("📁 Data Loading")
        uploaded_file = st.sidebar.file_uploader(
                "Upload your own CSV (optional):",
                type=['csv'],
                help="If no file is uploaded, titanic_clean.csv will be used by default"
        )
        
        # Load data
        default_file = "data/titanic_clean.csv"
        df = load_data(default_file, uploaded_file)
        
        if df.empty:
                st.error("❌ Could not load data. Verify that 'titanic_clean.csv' exists or upload your file.")
                return
        
        # Create filters in sidebar
        filters = create_filters(df)
        
        # Apply filters
        df_filtered = apply_filters(df, filters)
        
        # Check if filtered DataFrame is empty
        if df_filtered.empty:
                st.warning("⚠️ No data matches the selected filters. Adjust the filters.")
                return
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 EDA", "📈 Metrics", "🎨 Custom Plots"])
        
        # TAB 1: DATA
        with tab1:
                st.header("📊 Data View")
                
                # Controls for preview
                col1, col2 = st.columns(2)
                with col1:
                        n_rows = st.slider("Number of rows to show:", 1, min(100, len(df_filtered)), 10)
                with col2:
                        show_info = st.checkbox("Show dataset information")
                
                # Show data
                st.subheader(f"First {n_rows} rows (Total: {len(df_filtered)} records)")
                st.dataframe(df_filtered.head(n_rows), use_container_width=True)
                
                # Additional information
                if show_info:
                        col1, col2 = st.columns(2)
                        with col1:
                                st.subheader("📏 Dimensions")
                                st.write(f"Rows: {df_filtered.shape[0]}")
                                st.write(f"Columns: {df_filtered.shape[1]}")
                        
                        with col2:
                                st.subheader("🔧 Data Types")
                                st.write(df_filtered.dtypes)
                
                # Download button
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                        label="📥 Download filtered data (CSV)",
                        data=csv_data,
                        file_name="titanic_filtered.csv",
                        mime="text/csv",
                        use_container_width=True
                )
        
        # TAB 2: EDA
        with tab2:
                st.header("🔍 Exploratory Analysis")
                
                # Descriptive statistics
                st.subheader("📊 Descriptive Statistics")
                st.dataframe(df_filtered.describe(), use_container_width=True)
                
                # Main plots
                st.subheader("📈 Main Visualizations")
                plots = create_eda_plots(df_filtered)
                
                # Organize plots in columns
                if plots:
                        # First row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                                if 'age_hist' in plots:
                                        st.plotly_chart(plots['age_hist'], use_container_width=True)
                        
                        with col2:
                                if 'survival_by_sex' in plots:
                                        st.plotly_chart(plots['survival_by_sex'], use_container_width=True)
                        
                        # Second row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                                if 'survival_by_class' in plots:
                                        st.plotly_chart(plots['survival_by_class'], use_container_width=True)
                        
                        with col2:
                                if 'fare_by_class' in plots:
                                        st.plotly_chart(plots['fare_by_class'], use_container_width=True)
                        
                        # Third row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                                if 'fare_by_embarked' in plots:
                                        st.plotly_chart(plots['fare_by_embarked'], use_container_width=True)
                        
                        with col2:
                                if 'correlation_heatmap' in plots:
                                        st.plotly_chart(plots['correlation_heatmap'], use_container_width=True)
        
        # TAB 3: METRICS
        with tab3:
                st.header("📈 Key Metrics")
                
                # Create metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                        total_passengers = len(df_filtered)
                        st.metric("👥 Total Passengers", f"{total_passengers:,}")
                
                with col2:
                        if 'survived' in df_filtered.columns:
                                survival_rate = (df_filtered['survived'].sum() / len(df_filtered) * 100)
                                st.metric("💚 Survival Rate", f"{survival_rate:.1f}%")
                        else:
                                st.metric("💚 Survival Rate", "N/A")
                
                with col3:
                        if 'age' in df_filtered.columns:
                                avg_age = df_filtered['age'].mean()
                                st.metric("🎂 Average Age", f"{avg_age:.1f} years")
                        else:
                                st.metric("🎂 Average Age", "N/A")
                
                with col4:
                        if 'fare' in df_filtered.columns:
                                avg_fare = df_filtered['fare'].mean()
                                st.metric("💰 Average Fare", f"${avg_fare:.2f}")
                        else:
                                st.metric("💰 Average Fare", "N/A")
                
                # Additional metrics
                st.subheader("📊 Detailed Metrics")
                
                if 'sex' in df_filtered.columns:
                        sex_dist = df_filtered['sex'].value_counts()
                        col1, col2 = st.columns(2)
                        with col1:
                                st.write("**Distribution by Sex:**")
                                for sex, count in sex_dist.items():
                                        pct = (count / len(df_filtered) * 100)
                                        st.write(f"• {sex.title()}: {count} ({pct:.1f}%)")
                
                class_col = 'pclass' if 'pclass' in df_filtered.columns else 'class'
                if class_col in df_filtered.columns:
                        with col2:
                                st.write("**Distribution by Class:**")
                                class_dist = df_filtered[class_col].value_counts().sort_index()
                                for cls, count in class_dist.items():
                                        pct = (count / len(df_filtered) * 100)
                                        st.write(f"• Class {cls}: {count} ({pct:.1f}%)")
        
        # TAB 4: CUSTOM PLOTS
        with tab4:
                st.header("🎨 Custom Plots")
                
                # Controls for custom plots
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                        plot_type = st.selectbox(
                                "Plot Type:",
                                ['Histogram', 'Bar (count)', 'Boxplot', 'Scatter']
                        )
                
                with col2:
                        x_col = st.selectbox("X Variable:", df_filtered.columns.tolist())
                
                with col3:
                        if plot_type in ['Boxplot', 'Scatter']:
                                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                                y_col = st.selectbox("Y Variable:", [''] + numeric_cols)
                        else:
                                y_col = None
                
                with col4:
                        hue_options = [''] + [col for col in df_filtered.columns if df_filtered[col].nunique() <= 10]
                        hue_col = st.selectbox("Color/Grouping:", hue_options)
                        hue_col = hue_col if hue_col else None
                
                # Validate inputs and create plot
                if st.button("🚀 Generate Plot", type="primary"):
                        # Validations
                        if not x_col:
                                st.error("❌ You must select an X variable")
                        elif plot_type in ['Boxplot', 'Scatter'] and not y_col:
                                st.error("❌ For this plot type you need to select a numeric Y variable")
                        else:
                                # Create custom plot
                                custom_fig = create_custom_plot(df_filtered, x_col, y_col, plot_type, hue_col)
                                
                                if custom_fig:
                                        st.plotly_chart(custom_fig, use_container_width=True)
                                else:
                                        st.error("❌ Could not generate plot with selected variables")

if __name__ == "__main__":
        main()

# Final comments with instructions:
# Installation (example):
#   pip install streamlit pandas numpy plotly seaborn scikit-learn
# Execution:
#   streamlit run app.py
# The expected CSV by default is ./titanic_clean.csv (or upload your own CSV in the sidebar)