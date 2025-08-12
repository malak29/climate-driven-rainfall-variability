"""
Excel Data Processing Script for Indian Rainfall Analysis
This script processes your specific Excel files and generates analysis-ready datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RainfallDataProcessor:
    """
    Process the actual Excel files for rainfall analysis
    """
    
    def __init__(self):
        self.rainfall_data = None
        self.district_data = None
        self.processed_data = None
        self.summary_stats = {}
        
    def load_excel_files(self, rainfall_file='rainfall_in_india_1901-2015.xlsx', 
                        district_file='district_wise_rainfall_normal.xlsx'):
        """
        Load the Excel files
        """
        print("="*80)
        print("LOADING EXCEL FILES")
        print("="*80)
        
        try:
            # Load rainfall data
            print(f"\n📁 Loading: {rainfall_file}")
            self.rainfall_data = pd.read_excel(rainfall_file)
            print(f"   ✓ Loaded {len(self.rainfall_data)} rows")
            print(f"   ✓ Columns: {list(self.rainfall_data.columns)[:10]}...")
            
            # Load district data
            print(f"\n📁 Loading: {district_file}")
            self.district_data = pd.read_excel(district_file)
            print(f"   ✓ Loaded {len(self.district_data)} rows")
            print(f"   ✓ Columns: {list(self.district_data.columns)[:10]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return False
    
    def explore_data_structure(self):
        """
        Explore and understand the structure of loaded data
        """
        print("\n" + "="*80)
        print("DATA STRUCTURE ANALYSIS")
        print("="*80)
        
        # Rainfall data structure
        print("\n📊 RAINFALL DATA STRUCTURE:")
        print(f"   Shape: {self.rainfall_data.shape}")
        print(f"   Data Types:\n{self.rainfall_data.dtypes.value_counts()}")
        print(f"\n   First 5 rows:")
        print(self.rainfall_data.head())
        
        # Check for time columns
        time_cols = ['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        available_time_cols = [col for col in time_cols if col in self.rainfall_data.columns]
        print(f"\n   Time columns found: {available_time_cols}")
        
        # Check for location columns
        location_cols = ['STATE', 'DISTRICT', 'SUBDIVISION', 'REGION']
        available_location_cols = [col for col in location_cols if col in self.rainfall_data.columns]
        print(f"   Location columns found: {available_location_cols}")
        
        # District data structure
        print("\n📊 DISTRICT DATA STRUCTURE:")
        print(f"   Shape: {self.district_data.shape}")
        print(f"   First 5 rows:")
        print(self.district_data.head())
        
        return available_time_cols, available_location_cols
    
    def process_rainfall_data(self):
        """
        Process and transform the rainfall data for analysis
        """
        print("\n" + "="*80)
        print("PROCESSING RAINFALL DATA")
        print("="*80)
        
        # Identify columns
        time_cols, location_cols = self.explore_data_structure()
        
        # Create a processed dataset
        processed_data = []
        
        # Process based on available columns
        if 'YEAR' in self.rainfall_data.columns:
            print("\n🔄 Processing yearly data...")
            
            # Get month columns
            month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            available_months = [col for col in month_cols if col in self.rainfall_data.columns]
            
            # Calculate annual totals
            if available_months:
                self.rainfall_data['ANNUAL_TOTAL'] = self.rainfall_data[available_months].sum(axis=1)
                self.rainfall_data['MONSOON_TOTAL'] = self.rainfall_data[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1) if all(m in available_months for m in ['JUN', 'JUL', 'AUG', 'SEP']) else 0
                
                print(f"   ✓ Calculated annual totals")
                print(f"   ✓ Calculated monsoon totals")
        
        # Handle missing values
        print("\n🔄 Handling missing values...")
        missing_before = self.rainfall_data.isnull().sum().sum()
        self.rainfall_data = self.rainfall_data.interpolate(method='linear', limit_direction='both')
        missing_after = self.rainfall_data.isnull().sum().sum()
        print(f"   ✓ Missing values reduced from {missing_before} to {missing_after}")
        
        self.processed_data = self.rainfall_data
        return self.processed_data
    
    def calculate_statistics(self):
        """
        Calculate comprehensive statistics
        """
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        stats = {}
        
        # Overall statistics
        if 'ANNUAL_TOTAL' in self.processed_data.columns:
            stats['mean_annual_rainfall'] = self.processed_data['ANNUAL_TOTAL'].mean()
            stats['std_annual_rainfall'] = self.processed_data['ANNUAL_TOTAL'].std()
            stats['cv_rainfall'] = (stats['std_annual_rainfall'] / stats['mean_annual_rainfall']) * 100
            
            print(f"\n📈 OVERALL STATISTICS:")
            print(f"   Mean Annual Rainfall: {stats['mean_annual_rainfall']:.2f} mm")
            print(f"   Standard Deviation: {stats['std_annual_rainfall']:.2f} mm")
            print(f"   Coefficient of Variation: {stats['cv_rainfall']:.2f}%")
        
        # Trend analysis
        if 'YEAR' in self.processed_data.columns and 'ANNUAL_TOTAL' in self.processed_data.columns:
            from scipy import stats as scipy_stats
            
            yearly_avg = self.processed_data.groupby('YEAR')['ANNUAL_TOTAL'].mean()
            years = np.array(range(len(yearly_avg)))
            rainfall = yearly_avg.values
            
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, rainfall)
            
            stats['trend_slope'] = slope
            stats['trend_pvalue'] = p_value
            stats['trend_r2'] = r_value**2
            
            print(f"\n📊 TREND ANALYSIS:")
            print(f"   Trend: {'Decreasing' if slope < 0 else 'Increasing'}")
            print(f"   Rate of change: {slope:.2f} mm/year")
            print(f"   Statistical significance (p-value): {p_value:.4f}")
            print(f"   R-squared: {stats['trend_r2']:.3f}")
        
        # Regional analysis
        if any(col in self.processed_data.columns for col in ['STATE', 'DISTRICT', 'SUBDIVISION']):
            location_col = next((col for col in ['STATE', 'DISTRICT', 'SUBDIVISION'] 
                                if col in self.processed_data.columns), None)
            
            if location_col and 'ANNUAL_TOTAL' in self.processed_data.columns:
                regional_stats = self.processed_data.groupby(location_col)['ANNUAL_TOTAL'].agg(['mean', 'std', 'min', 'max'])
                
                print(f"\n🗺️ REGIONAL ANALYSIS (by {location_col}):")
                print(f"   Top 5 wettest regions:")
                top_5 = regional_stats.nlargest(5, 'mean')
                for region, row in top_5.iterrows():
                    print(f"      {region}: {row['mean']:.0f} mm (σ={row['std']:.0f})")
                
                print(f"\n   Top 5 driest regions:")
                bottom_5 = regional_stats.nsmallest(5, 'mean')
                for region, row in bottom_5.iterrows():
                    print(f"      {region}: {row['mean']:.0f} mm (σ={row['std']:.0f})")
                
                stats['regional_stats'] = regional_stats.to_dict()
        
        self.summary_stats = stats
        return stats
    
    def identify_extremes(self, threshold_percentile=5):
        """
        Identify extreme rainfall events
        """
        print("\n" + "="*80)
        print("EXTREME EVENT ANALYSIS")
        print("="*80)
        
        extremes = {}
        
        if 'ANNUAL_TOTAL' in self.processed_data.columns:
            # Define thresholds
            drought_threshold = self.processed_data['ANNUAL_TOTAL'].quantile(threshold_percentile/100)
            flood_threshold = self.processed_data['ANNUAL_TOTAL'].quantile(1 - threshold_percentile/100)
            
            # Identify extreme years
            if 'YEAR' in self.processed_data.columns:
                drought_years = self.processed_data[self.processed_data['ANNUAL_TOTAL'] < drought_threshold]['YEAR'].unique()
                flood_years = self.processed_data[self.processed_data['ANNUAL_TOTAL'] > flood_threshold]['YEAR'].unique()
                
                print(f"\n🌵 DROUGHT EVENTS (Bottom {threshold_percentile}%):")
                print(f"   Threshold: < {drought_threshold:.0f} mm")
                print(f"   Years: {sorted(drought_years)[:10]}...")
                print(f"   Total occurrences: {len(drought_years)}")
                
                print(f"\n💧 EXTREME RAINFALL EVENTS (Top {threshold_percentile}%):")
                print(f"   Threshold: > {flood_threshold:.0f} mm")
                print(f"   Years: {sorted(flood_years)[:10]}...")
                print(f"   Total occurrences: {len(flood_years)}")
                
                extremes['drought_years'] = list(drought_years)
                extremes['flood_years'] = list(flood_years)
            
            extremes['drought_threshold'] = drought_threshold
            extremes['flood_threshold'] = flood_threshold
        
        return extremes
    
    def create_visualizations(self, save_path='./'):
        """
        Create comprehensive visualizations
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Indian Rainfall Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        if 'YEAR' in self.processed_data.columns and 'ANNUAL_TOTAL' in self.processed_data.columns:
            yearly_data = self.processed_data.groupby('YEAR')['ANNUAL_TOTAL'].mean()
            axes[0, 0].plot(yearly_data.index, yearly_data.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Annual Rainfall Trend (1901-2015)')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Rainfall (mm)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(yearly_data)), yearly_data.values, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(yearly_data.index, p(range(len(yearly_data))), "r--", alpha=0.8, label=f'Trend')
            axes[0, 0].legend()
        
        # 2. Monthly distribution
        month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        available_months = [col for col in month_cols if col in self.processed_data.columns]
        if available_months:
            monthly_avg = self.processed_data[available_months].mean()
            axes[0, 1].bar(range(len(monthly_avg)), monthly_avg.values, color='skyblue', edgecolor='navy')
            axes[0, 1].set_title('Average Monthly Rainfall Distribution')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Rainfall (mm)')
            axes[0, 1].set_xticks(range(len(monthly_avg)))
            axes[0, 1].set_xticklabels(monthly_avg.index, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Regional comparison
        if any(col in self.processed_data.columns for col in ['STATE', 'DISTRICT', 'SUBDIVISION']):
            location_col = next((col for col in ['STATE', 'DISTRICT', 'SUBDIVISION'] 
                                if col in self.processed_data.columns), None)
            if location_col and 'ANNUAL_TOTAL' in self.processed_data.columns:
                top_regions = self.processed_data.groupby(location_col)['ANNUAL_TOTAL'].mean().nlargest(10)
                axes[0, 2].barh(range(len(top_regions)), top_regions.values, color='green', alpha=0.7)
                axes[0, 2].set_title(f'Top 10 Regions by Rainfall')
                axes[0, 2].set_xlabel('Average Annual Rainfall (mm)')
                axes[0, 2].set_yticks(range(len(top_regions)))
                axes[0, 2].set_yticklabels(top_regions.index, fontsize=8)
                axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Coefficient of Variation
        if 'ANNUAL_TOTAL' in self.processed_data.columns:
            axes[1, 0].hist(self.processed_data['ANNUAL_TOTAL'].dropna(), bins=50, color='coral', edgecolor='darkred', alpha=0.7)
            axes[1, 0].set_title('Rainfall Distribution Histogram')
            axes[1, 0].set_xlabel('Annual Rainfall (mm)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(self.processed_data['ANNUAL_TOTAL'].mean(), color='red', linestyle='--', label='Mean')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monsoon vs Non-Monsoon
        if 'MONSOON_TOTAL' in self.processed_data.columns and 'ANNUAL_TOTAL' in self.processed_data.columns:
            monsoon_pct = (self.processed_data['MONSOON_TOTAL'] / self.processed_data['ANNUAL_TOTAL'] * 100).mean()
            sizes = [monsoon_pct, 100 - monsoon_pct]
            labels = [f'Monsoon\n({monsoon_pct:.1f}%)', f'Non-Monsoon\n({100-monsoon_pct:.1f}%)']
            colors = ['#1f77b4', '#ff7f0e']
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Monsoon vs Non-Monsoon Rainfall Contribution')
        
        # 6. Decadal comparison
        if 'YEAR' in self.processed_data.columns and 'ANNUAL_TOTAL' in self.processed_data.columns:
            self.processed_data['DECADE'] = (self.processed_data['YEAR'] // 10) * 10
            decadal_avg = self.processed_data.groupby('DECADE')['ANNUAL_TOTAL'].mean()
            axes[1, 2].plot(decadal_avg.index, decadal_avg.values, 'go-', linewidth=2, markersize=8)
            axes[1, 2].set_title('Decadal Average Rainfall')
            axes[1, 2].set_xlabel('Decade')
            axes[1, 2].set_ylabel('Average Rainfall (mm)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{save_path}rainfall_analysis_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved visualization: {filename}")
        
        plt.show()
        
        return fig
    
    def export_processed_data(self, output_file='processed_rainfall_data.csv'):
        """
        Export processed data for further analysis
        """
        print("\n" + "="*80)
        print("EXPORTING PROCESSED DATA")
        print("="*80)
        
        if self.processed_data is not None:
            self.processed_data.to_csv(output_file, index=False)
            print(f"   ✓ Exported to: {output_file}")
            print(f"   ✓ Shape: {self.processed_data.shape}")
            
            # Also export summary statistics
            stats_file = output_file.replace('.csv', '_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(self.summary_stats, f, indent=4, default=str)
            print(f"   ✓ Statistics exported to: {stats_file}")
        else:
            print("   ❌ No processed data to export")
    
    def generate_report(self):
        """
        Generate a comprehensive text report
        """
        print("\n" + "="*80)
        print("GENERATING ANALYSIS REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("INDIAN RAINFALL ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Data Overview
        report.append("\n1. DATA OVERVIEW")
        report.append("-" * 40)
        if self.processed_data is not None:
            report.append(f"   Total Records: {len(self.processed_data):,}")
            report.append(f"   Columns: {len(self.processed_data.columns)}")
            
            if 'YEAR' in self.processed_data.columns:
                report.append(f"   Time Period: {self.processed_data['YEAR'].min()} - {self.processed_data['YEAR'].max()}")
        
        # Key Statistics
        report.append("\n2. KEY STATISTICS")
        report.append("-" * 40)
        if self.summary_stats:
            if 'mean_annual_rainfall' in self.summary_stats:
                report.append(f"   Mean Annual Rainfall: {self.summary_stats['mean_annual_rainfall']:.2f} mm")
                report.append(f"   Standard Deviation: {self.summary_stats['std_annual_rainfall']:.2f} mm")
                report.append(f"   Coefficient of Variation: {self.summary_stats['cv_rainfall']:.2f}%")
            
            if 'trend_slope' in self.summary_stats:
                trend = "Decreasing" if self.summary_stats['trend_slope'] < 0 else "Increasing"
                report.append(f"   Long-term Trend: {trend}")
                report.append(f"   Rate of Change: {abs(self.summary_stats['trend_slope']):.2f} mm/year")
        
        # Recommendations
        report.append("\n3. KEY FINDINGS & RECOMMENDATIONS")
        report.append("-" * 40)
        
        if self.summary_stats.get('cv_rainfall', 0) > 20:
            report.append("   ⚠️ High rainfall variability detected (CV > 20%)")
            report.append("      → Implement robust water storage infrastructure")
            report.append("      → Develop drought contingency plans")
        
        if self.summary_stats.get('trend_slope', 0) < 0:
            report.append("   ⚠️ Declining rainfall trend observed")
            report.append("      → Promote water conservation measures")
            report.append("      → Invest in rainwater harvesting")
        
        report.append("\n4. DISTRICT-LEVEL ACTIONS NEEDED")
        report.append("-" * 40)
        report.append("   • Immediate: Deploy early warning systems in high-risk districts")
        report.append("   • Short-term: Expand crop insurance coverage to vulnerable farmers")
        report.append("   • Long-term: Develop climate-resilient agricultural practices")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('rainfall_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n   ✓ Report saved to: rainfall_analysis_report.txt")
        
        return report_text

# Main execution
if __name__ == "__main__":
    print("\n" + "🌧️"*20)
    print("     INDIAN RAINFALL DATA PROCESSING SYSTEM")
    print("🌧️"*20 + "\n")
    
    # Initialize processor
    processor = RainfallDataProcessor()
    
    # Load Excel files
    if processor.load_excel_files():
        # Process data
        processor.process_rainfall_data()
        
        # Calculate statistics
        processor.calculate_statistics()
        
        # Identify extremes
        processor.identify_extremes()
        
        # Create visualizations
        processor.create_visualizations()
        
        # Export processed data
        processor.export_processed_data()
        
        # Generate report
        processor.generate_report()
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  1. rainfall_analysis_dashboard.png - Visual dashboard")
        print("  2. processed_rainfall_data.csv - Processed dataset")
        print("  3. processed_rainfall_data_statistics.json - Statistical summary")
        print("  4. rainfall_analysis_report.txt - Analysis report")
        print("\n🚀 Ready for advanced predictive modeling and dashboard deployment!")
    else:
        print("\n❌ Failed to load Excel files. Please check file paths and try again.")