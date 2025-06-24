import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openai import AzureOpenAI
from config import Config
from advanced_data_manager import AdvancedDataManager, TimeRange, BudgetInfo
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import random
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced FinOps AI Agent - Multi-Cloud Cost Management",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .budget-exceeded {
        background-color: #ffe6e6;
        border-left-color: #d32f2f;
    }
    .budget-warning {
        background-color: #fff3e0;
        border-left-color: #f57c00;
    }
    .budget-ok {
        background-color: #e8f5e8;
        border-left-color: #388e3c;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .alert-danger { 
        background-color: #f8d7da; 
        border-left-color: #dc3545;
        color: #721c24;
    }
    .alert-warning { 
        background-color: #fff3cd; 
        border-left-color: #ffc107;
        color: #856404;
    }
    .alert-success { 
        background-color: #d4edda; 
        border-left-color: #28a745;
        color: #155724;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    .cloud-selector {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Azure OpenAI client
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_version=Config.AZURE_API_VERSION,
        azure_endpoint=Config.AZURE_ENDPOINT,
        api_key=Config.AZURE_API_KEY,
    )

# Initialize enhanced data manager
@st.cache_data
def load_enhanced_data():
    return AdvancedDataManager()

# Session state initialization for alerts and spending limits
def init_session_state():
    if 'spending_limits' not in st.session_state:
        st.session_state.spending_limits = {
            'AWS': {'limit': 5000, 'enabled': True},
            'Azure': {'limit': 3000, 'enabled': True},
            'GCP': {'limit': 2000, 'enabled': True}
        }
    if 'alert_settings' not in st.session_state:
        st.session_state.alert_settings = {
            'untagged_resources': True,
            'unused_resources': True,
            'budget_warnings': True,
            'spending_threshold': 80  # Alert at 80% of spending limit
        }

def create_enhanced_system_message(data_manager, selected_clouds):
    """Create comprehensive system message with multi-cloud cost insights"""
    cost_by_provider = data_manager.get_cost_summary_by_provider()
    cost_by_service = data_manager.get_cost_summary_by_service()
    insights = data_manager.get_optimization_insights()
    budgets = data_manager.get_budget_status()
    audit = data_manager.perform_finops_audit()
    
    # Filter by selected clouds
    if selected_clouds and "All Clouds" not in selected_clouds:
        filtered_costs = {k: v for k, v in cost_by_provider.items() if k in selected_clouds}
    else:
        filtered_costs = cost_by_provider
    
    cost_summary_str = "\n".join([f"{prov}: ${amt:.2f}" for prov, amt in filtered_costs.items()])
    service_summary_str = "\n".join([f"{svc}: ${amt:.2f}" for svc, amt in sorted(cost_by_service.items(), key=lambda x: x[1], reverse=True)[:10]])
    insights_str = "\n".join(insights)
    budget_str = "\n".join([f"{b.name}: ${b.actual_spend:.2f}/${b.limit:.2f} ({b.status})" for b in budgets])
    
    return f"""
You are an advanced FinOps (Financial Operations) AI agent created by lab7ai to assist customers in managing and optimizing their multi-cloud costs across AWS, Azure, and GCP. You have enhanced capabilities for unified cloud cost management, audit reporting, and automated alerting.

STRICT GUARD RAILS - CRITICAL INSTRUCTIONS:
🚫 SCOPE LIMITATION: You MUST ONLY respond to queries related to:
- Cloud cost management and optimization (AWS, Azure, GCP)
- FinOps practices and methodologies
- Cloud billing, budgets, and financial planning
- Resource optimization and rightsizing
- Cloud governance and cost allocation
- Multi-cloud financial operations
- Cloud cost monitoring and alerting
- Resource tagging and management
- Cloud spending analysis and forecasting

🚫 REFUSE ALL OTHER TOPICS: If asked about ANYTHING outside the scope above (like biology, general knowledge, personal advice, other technologies not related to cloud costs, etc.), you MUST:
1. Politely refuse to answer
2. Remind the user of your specific purpose
3. Suggest relevant FinOps questions they could ask instead

Example refusal response:
"I'm sorry, but I can only assist with cloud FinOps and cost management topics across AWS, Azure, and GCP. I'm designed specifically to help with cloud cost optimization, budget management, and financial operations.

Instead, you could ask me about:
- 'How can I optimize my AWS EC2 costs?'
- 'Show me my multi-cloud spending trends'
- 'What are the best practices for cloud cost allocation?'
- 'How can I set up automated cost alerts?'
- 'Analyze my untagged resources and their costs'"

IMPORTANT: If someone asks who created you, respond: "I was created by lab7ai to assist customers in managing their cloud costs across AWS, Azure, and GCP, with advanced FinOps capabilities for unified cost management and optimization."

**Current Multi-Cloud Financial Overview (Selected Clouds: {', '.join(selected_clouds) if selected_clouds else 'All'}):**

**Total Cost by Provider:**
{cost_summary_str}

**Top Services by Cost:**
{service_summary_str}

**Budget Status:**
{budget_str}

**Current Optimization Insights:**
{insights_str}

**Audit Summary:**
- Untagged Resources: {len(audit.untagged_resources)}
- Stopped Instances: {len(audit.stopped_instances)}
- Unused Volumes: {len(audit.unused_volumes)}
- Budget Alerts: {len(audit.budget_alerts)}

**Your Enhanced FinOps Capabilities:**
✅ Multi-cloud unified cost management (AWS, Azure, GCP)
✅ Cloud-specific audit reports (JSON, CSV, PDF formats)
✅ 6-month cost trend analysis and forecasting
✅ Automated spending limit alerts and notifications
✅ Untagged resource identification and cost allocation
✅ Underutilized resource detection and optimization
✅ Savings plan recommendations and tracking
✅ Comprehensive report generation (Cost, Trend, Audit)
✅ Real-time budget monitoring and alerting
✅ Cross-cloud resource consolidation and analysis
✅ **NEW**: Interactive resource tagging via chat commands
✅ **NEW**: Interactive stopped instance management
✅ **NEW**: Interactive unused volume cleanup
✅ **NEW**: Project-based cost analysis and visualization
✅ **NEW**: AI-powered cost predictions with confidence intervals
✅ **NEW**: Intelligent anomaly detection for cost patterns
✅ **NEW**: ML-driven optimization recommendations
✅ **NEW**: Smart tagging suggestions with confidence scoring
✅ **NEW**: Contextual priority alerts and notifications

**Interactive Commands Available:**
🏷️ **Tagging Commands:**
- "Tag resource [resource_name] with Project=ProjectName,Environment=Production"
- "Tag all untagged EC2 resources with Project=WebApp,Team=DevOps"
- "Show untagged resources"

⏹️ **Instance Management:**
- "Terminate stopped instance [instance_id]"
- "Start stopped instance [instance_id]"
- "Show stopped instances"

💾 **Volume Management:**
- "Delete unused volume [volume_id]"
- "Attach volume [volume_id] to instance [instance_id]"
- "Show unused volumes"

📊 **Project Analysis:**
- "Show project costs"
- "Show costs for project [ProjectName]"

**Guidelines:**
- Provide cloud-specific recommendations when users select specific providers
- Always highlight potential cost savings with specific dollar amounts
- Reference audit findings and suggest immediate actionable steps
- Recommend appropriate spending limits based on usage patterns
- Proactively suggest cost optimization opportunities
- Alert users to urgent budget or resource waste issues
- Assist with interactive resource management through chat commands
"""

def render_cloud_selector():
    """Render cloud provider selection interface"""
    st.markdown('<div class="cloud-selector">', unsafe_allow_html=True)
    st.subheader("☁️ Cloud Provider Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_clouds = st.multiselect(
            "Select Cloud Providers for Analysis",
            ["All Clouds", "AWS", "Azure", "GCP"],
            default=["All Clouds"],
            help="Choose specific cloud providers or select 'All Clouds' for unified analysis"
        )
    
    with col2:
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["Cost & Usage", "Audit Only", "Trends Only", "Comprehensive"],
            help="Choose the type of analysis to perform"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return selected_clouds, analysis_scope

def render_spending_limits_config():
    """Render spending limits configuration"""
    st.subheader("💰 Spending Limits & Alerts Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Monthly Spending Limits**")
        for cloud in ['AWS', 'Azure', 'GCP']:
            enabled = st.checkbox(f"Enable {cloud} limit", value=st.session_state.spending_limits[cloud]['enabled'], key=f"{cloud}_enabled")
            limit = st.number_input(f"{cloud} Monthly Limit ($)", 
                                  value=st.session_state.spending_limits[cloud]['limit'], 
                                  min_value=100, 
                                  step=100,
                                  key=f"{cloud}_limit")
            st.session_state.spending_limits[cloud] = {'limit': limit, 'enabled': enabled}
    
    with col2:
        st.write("**Alert Settings**")
        threshold = st.slider("Alert Threshold (%)", 50, 100, st.session_state.alert_settings['spending_threshold'])
        untagged_alerts = st.checkbox("Untagged Resource Alerts", value=st.session_state.alert_settings['untagged_resources'])
        unused_alerts = st.checkbox("Unused Resource Alerts", value=st.session_state.alert_settings['unused_resources'])
        budget_alerts = st.checkbox("Budget Warning Alerts", value=st.session_state.alert_settings['budget_warnings'])
        
        st.session_state.alert_settings = {
            'spending_threshold': threshold,
            'untagged_resources': untagged_alerts,
            'unused_resources': unused_alerts,
            'budget_warnings': budget_alerts
        }

def check_and_display_alerts(data_manager, selected_clouds):
    """Check for alerts and display notifications"""
    alerts = []
    
    # Check spending limits
    provider_costs = data_manager.get_cost_summary_by_provider()
    for cloud, settings in st.session_state.spending_limits.items():
        if settings['enabled'] and cloud in provider_costs:
            current_spend = provider_costs[cloud]
            limit = settings['limit']
            percentage = (current_spend / limit) * 100
            
            if percentage >= st.session_state.alert_settings['spending_threshold']:
                alert_type = "danger" if percentage >= 100 else "warning"
                alerts.append({
                    'type': alert_type,
                    'message': f"🚨 {cloud} spending: ${current_spend:,.2f} ({percentage:.1f}% of ${limit:,.2f} limit)"
                })
    
    # Check audit findings
    if st.session_state.alert_settings['untagged_resources']:
        audit = data_manager.perform_finops_audit()
        if audit.untagged_resources:
            alerts.append({
                'type': 'warning',
                'message': f"🏷️ Found {len(audit.untagged_resources)} untagged resources affecting cost allocation"
            })
    
    if st.session_state.alert_settings['unused_resources']:
        audit = data_manager.perform_finops_audit()
        if audit.unused_volumes:
            total_size = sum(vol['size'] for vol in audit.unused_volumes)
            estimated_savings = total_size * 0.10
            alerts.append({
                'type': 'info',
                'message': f"💾 Found {len(audit.unused_volumes)} unused volumes - Potential savings: ${estimated_savings:,.2f}/month"
            })
    
    # Display alerts
    if alerts:
        st.subheader("🔔 Active Alerts & Notifications")
        for alert in alerts:
            css_class = f"alert-{alert['type']}"
            st.markdown(f'<div class="alert-box {css_class}">{alert["message"]}</div>', unsafe_allow_html=True)

def render_unified_dashboard(data_manager, selected_clouds):
    """Render unified multi-cloud dashboard"""
    st.subheader("📊 Unified Multi-Cloud Dashboard")
    
    # Filter data by selected clouds
    provider_costs = data_manager.get_cost_summary_by_provider()
    if selected_clouds and "All Clouds" not in selected_clouds:
        provider_costs = {k: v for k, v in provider_costs.items() if k in selected_clouds}
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = sum(provider_costs.values()) if provider_costs else 0
    audit = data_manager.perform_finops_audit()
    budgets = data_manager.get_budget_status()
    
    with col1:
        st.metric("💰 Total Monthly Cost", f"${total_cost:,.2f}")
    
    with col2:
        active_clouds = len(provider_costs)
        st.metric("☁️ Active Clouds", active_clouds)
    
    with col3:
        optimization_items = len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes)
        st.metric("🔍 Optimization Items", optimization_items)
    
    with col4:
        budget_alerts = len([b for b in budgets if b.status in ['warning', 'exceeded']])
        st.metric("⚠️ Budget Alerts", budget_alerts)
    
    # NEW: Project Costs Section
    st.subheader("🏗️ Costs by Project")
    project_costs = data_manager.get_cost_by_project()
    
    if project_costs:
        # Project costs metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_projects = len(project_costs)
        highest_cost_project = max(project_costs.items(), key=lambda x: x[1]['total_cost'])
        untagged_cost = project_costs.get('Untagged', {}).get('total_cost', 0)
        
        with col1:
            st.metric("🏗️ Total Projects", total_projects)
        
        with col2:
            st.metric("📈 Highest Cost Project", highest_cost_project[0])
            st.caption(f"${highest_cost_project[1]['total_cost']:,.2f}")
        
        with col3:
            st.metric("🏷️ Untagged Resources Cost", f"${untagged_cost:,.2f}")
        
        with col4:
            total_tagged_cost = sum(data['total_cost'] for name, data in project_costs.items() if name != 'Untagged')
            st.metric("✅ Tagged Resources Cost", f"${total_tagged_cost:,.2f}")
        
        # Project costs visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Project costs pie chart
            project_names = list(project_costs.keys())
            project_values = [data['total_cost'] for data in project_costs.values()]
            
            fig_project_pie = px.pie(
                values=project_values,
                names=project_names,
                title="💰 Cost Distribution by Project",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_project_pie, use_container_width=True)
        
        with col2:
            # Top projects bar chart
            sorted_projects = sorted(project_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)[:8]
            project_df = pd.DataFrame([
                {'Project': name, 'Cost': data['total_cost'], 'Resources': data['resource_count']}
                for name, data in sorted_projects
            ])
            
            fig_project_bar = px.bar(
                project_df,
                x='Project',
                y='Cost',
                title="📊 Top Projects by Cost",
                color='Cost',
                color_continuous_scale='Blues',
                text='Resources'
            )
            fig_project_bar.update_layout(xaxis_tickangle=-45)
            fig_project_bar.update_traces(texttemplate='%{text} resources', textposition='outside')
            st.plotly_chart(fig_project_bar, use_container_width=True)
        
        # Project details table
        with st.expander("📋 Detailed Project Breakdown"):
            project_details = []
            for project_name, project_data in sorted_projects:
                project_details.append({
                    'Project': project_name,
                    'Total Cost': f"${project_data['total_cost']:,.2f}",
                    'Resource Count': project_data['resource_count'],
                    'Top Provider': max(project_data['providers'].items(), key=lambda x: x[1])[0] if project_data['providers'] else 'N/A',
                    'Top Service': max(project_data['services'].items(), key=lambda x: x[1])[0] if project_data['services'] else 'N/A'
                })
            
            project_df_detailed = pd.DataFrame(project_details)
            st.dataframe(project_df_detailed, use_container_width=True)
    else:
        st.info("No project cost data available. Ensure resources are properly tagged with Project=ProjectName.")
    
    # Cost distribution visualization
    if provider_costs:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=list(provider_costs.values()),
                names=list(provider_costs.keys()),
                title="💰 Cost Distribution by Cloud Provider",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Spending vs Limits comparison
            spending_data = []
            for cloud, cost in provider_costs.items():
                if cloud in st.session_state.spending_limits:
                    limit = st.session_state.spending_limits[cloud]['limit']
                    spending_data.append({
                        'Cloud': cloud,
                        'Current Spend': cost,
                        'Monthly Limit': limit,
                        'Remaining': max(0, limit - cost)
                    })
            
            if spending_data:
                spending_df = pd.DataFrame(spending_data)
                fig_bar = px.bar(
                    spending_df,
                    x='Cloud',
                    y=['Current Spend', 'Remaining'],
                    title="💳 Spending vs Limits",
                    barmode='stack',
                    color_discrete_map={'Current Spend': '#ff7f7f', 'Remaining': '#7fbf7f'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

def render_six_month_trends(data_manager, selected_clouds):
    """Render 6-month cost trend analysis"""
    st.subheader("📈 6-Month Cost Trend Analysis")
    
    trends_df = data_manager.get_cost_trends_six_months()
    
    if not trends_df.empty:
        # Filter by selected clouds
        if selected_clouds and "All Clouds" not in selected_clouds:
            trends_df = trends_df[trends_df['provider'].isin(selected_clouds)]
        
        if not trends_df.empty:
            # Line chart for trends
            fig_line = px.line(
                trends_df,
                x='month',
                y='cost_usd',
                color='provider',
                title="📊 Monthly Cost Trends (Last 6 Months)",
                markers=True,
                labels={'cost_usd': 'Cost (USD)', 'month': 'Month'}
            )
            fig_line.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Trend summary with improved error handling
            col1, col2, col3 = st.columns(3)
            
            # Group by month and calculate safely
            monthly_totals = trends_df.groupby('month')['cost_usd'].sum()
            
            if len(monthly_totals) >= 1:
                current_month_cost = monthly_totals.iloc[-1]
                col1.metric("📅 Current Month", f"${current_month_cost:,.2f}")
            else:
                col1.metric("📅 Current Month", "No data")
                
            if len(monthly_totals) >= 2:
                previous_month_cost = monthly_totals.iloc[-2]
                trend_change = ((current_month_cost - previous_month_cost) / previous_month_cost * 100) if previous_month_cost > 0 else 0
                col2.metric("📆 Previous Month", f"${previous_month_cost:,.2f}")
                col3.metric("📈 Month-over-Month", f"{trend_change:+.1f}%")
            else:
                col2.metric("📆 Previous Month", "Insufficient data")
                col3.metric("📈 Month-over-Month", "N/A")
        else:
            st.info("No trend data available for selected cloud providers.")
    else:
        st.info("No 6-month trend data available.")

def render_cloud_audit_report(data_manager, selected_clouds):
    """Render cloud-specific audit report"""
    st.subheader("🔍 Cloud Account Audit Report")
    
    audit = data_manager.perform_finops_audit()
    
    # Filter audit data by selected clouds if applicable
    if selected_clouds and "All Clouds" not in selected_clouds:
        # Filter untagged resources by provider
        filtered_untagged = []
        for resource in audit.untagged_resources:
            if 'provider' in resource and resource['provider'] in selected_clouds:
                filtered_untagged.append(resource)
        audit.untagged_resources = filtered_untagged
    
    # Audit overview
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("🏷️ Untagged Resources", len(audit.untagged_resources))
    col2.metric("⏹️ Stopped Instances", len(audit.stopped_instances))
    col3.metric("💾 Unused Volumes", len(audit.unused_volumes))
    col4.metric("🚨 Budget Issues", len(audit.budget_alerts))
    
    # Detailed audit tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Untagged Resources", "⏹️ Idle Resources", "💾 Waste Detection", "💰 Savings Opportunities"])
    
    with tab1:
        if audit.untagged_resources:
            st.warning(f"Found {len(audit.untagged_resources)} untagged resources affecting cost allocation")
            untagged_df = pd.DataFrame(audit.untagged_resources)
            if 'cost_usd' in untagged_df.columns:
                total_untagged_cost = untagged_df['cost_usd'].sum()
                st.metric("💰 Cost of Untagged Resources", f"${total_untagged_cost:,.2f}")
            st.dataframe(untagged_df, use_container_width=True)
            
            # Interactive tagging buttons
            st.write("**🎮 Interactive Tagging:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Show Detailed Untagged Resources", key="show_untagged"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Show untagged resources"})
                    st.rerun()
            
            with col2:
                if st.button("🏷️ Get Tagging Recommendations", key="tag_rec"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Recommend tagging strategy for untagged resources"})
                    st.rerun()
            
            st.info("💡 Use chat commands like 'Tag resource [resource_name] with Project=ProjectName,Environment=Production' to tag resources.")
        else:
            st.success("✅ All resources are properly tagged!")
    
    with tab2:
        if audit.stopped_instances:
            st.warning(f"Found {len(audit.stopped_instances)} stopped instances")
            stopped_df = pd.DataFrame(audit.stopped_instances)
            st.dataframe(stopped_df, use_container_width=True)
            
            # Interactive management buttons
            st.write("**🎮 Interactive Management:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Show Detailed Stopped Instances", key="show_stopped"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Show stopped instances"})
                    st.rerun()
            
            with col2:
                if st.button("⚡ Get Termination Recommendations", key="term_rec"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Which stopped instances should I terminate for cost optimization?"})
                    st.rerun()
            
            st.info("💡 Use chat commands like 'Terminate stopped instance [instance_id]' to manage instances.")
        else:
            st.success("✅ No idle instances detected!")
    
    with tab3:
        if audit.unused_volumes:
            total_size = sum(vol['size'] for vol in audit.unused_volumes)
            estimated_savings = total_size * 0.10
            st.error(f"Found {len(audit.unused_volumes)} unused volumes")
            st.metric("💸 Potential Monthly Savings", f"${estimated_savings:,.2f}")
            volumes_df = pd.DataFrame(audit.unused_volumes)
            st.dataframe(volumes_df, use_container_width=True)
            
            # Interactive management buttons
            st.write("**🎮 Interactive Management:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Show Detailed Unused Volumes", key="show_volumes"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Show unused volumes"})
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Get Cleanup Recommendations", key="cleanup_rec"):
                    st.session_state.enhanced_messages.append({"role": "user", "content": "Which unused volumes should I delete for cost savings?"})
                    st.rerun()
            
            st.info("💡 Use chat commands like 'Delete unused volume [volume_id]' to manage volumes.")
        else:
            st.success("✅ No unused volumes detected!")
    
    with tab4:
        insights = data_manager.get_optimization_insights()
        st.write("**Immediate Optimization Opportunities:**")
        for i, insight in enumerate(insights, 1):
            st.write(f"**{i}.** {insight}")

def render_comprehensive_reporting(data_manager, selected_clouds):
    """Enhanced comprehensive reporting with PDF generation"""
    st.subheader("📊 Enhanced Report Generation")
    st.write("Generate comprehensive reports with detailed analysis, charts, and actionable insights.")
    
    # Report configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Report Configuration")
        
        # Report types selection
        report_types = st.multiselect(
            "Select Report Sections",
            ["executive_summary", "cost_analysis", "resource_audit", "trend_analysis", "budget_analysis"],
            default=["executive_summary", "cost_analysis", "resource_audit"],
            help="Choose which sections to include in your report"
        )
        
        # Report format selection
        report_format = st.radio(
            "Report Format",
            ["PDF", "Excel", "JSON", "CSV"],
            index=0,
            help="Choose the format for your generated report"
        )
        
        # Additional options
        include_charts = st.checkbox("Include visualizations", value=True)
        include_recommendations = st.checkbox("Include actionable recommendations", value=True)
        include_savings_analysis = st.checkbox("Include potential savings analysis", value=True)
        
        # Report scope
        report_scope = st.selectbox(
            "Report Scope",
            ["Last 30 days", "Last 90 days", "Last 6 months", "Current month", "Custom range"],
            index=1
        )
    
    with col2:
        st.subheader("📈 Report Preview")
        
        # Show preview metrics
        provider_costs = data_manager.get_cost_summary_by_provider()
        total_cost = sum(provider_costs.get(cloud, 0) for cloud in selected_clouds)
        audit = data_manager.perform_finops_audit()
        
        st.metric("Total Cost", f"${total_cost:,.2f}")
        st.metric("Optimization Items", len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes))
        st.metric("Potential Savings", f"${(len(audit.stopped_instances) * 50 + len(audit.unused_volumes) * 25):,.2f}/month")
        
        # Cloud provider breakdown
        st.write("**Provider Breakdown:**")
        for cloud in selected_clouds:
            if cloud in provider_costs and provider_costs[cloud] > 0:
                percentage = (provider_costs[cloud] / total_cost * 100) if total_cost > 0 else 0
                st.write(f"• {cloud}: ${provider_costs[cloud]:,.2f} ({percentage:.1f}%)")
    
    st.divider()
    
    # Report generation section
    st.subheader("🚀 Generate Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report", type="primary"):
            if report_types:
                with st.spinner("Generating comprehensive PDF report..."):
                    try:
                        pdf_data = generate_pdf_report(data_manager, selected_clouds, report_types)
                        if pdf_data:
                            # Create filename with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"finops_report_{timestamp}.pdf"
                            
                            # Create download button
                            download_link = create_download_button(pdf_data, filename)
                            if download_link:
                                st.markdown(download_link, unsafe_allow_html=True)
                                st.success("✅ PDF report generated successfully!")
                                
                                # Show report summary
                                st.info(f"""
                                📋 **Report Summary:**
                                • Format: PDF
                                • Sections: {len(report_types)}
                                • Cloud Providers: {len(selected_clouds)}
                                • Total Pages: ~{len(report_types) * 2 + 3}
                                • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                """)
                        else:
                            st.error("❌ Failed to generate PDF report")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one report section")
    
    with col2:
        if st.button("📊 Generate Excel Report"):
            if report_types:
                with st.spinner("Generating Excel report..."):
                    try:
                        # Generate Excel report
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"finops_excel_report_{timestamp}.xlsx"
                        
                        # Create Excel data
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            # Cost summary sheet
                            cost_data = []
                            for cloud in selected_clouds:
                                if cloud in provider_costs:
                                    cost_data.append({
                                        'Cloud Provider': cloud,
                                        'Monthly Cost': provider_costs[cloud],
                                        'Percentage': (provider_costs[cloud] / total_cost * 100) if total_cost > 0 else 0
                                    })
                            
                            if cost_data:
                                cost_df = pd.DataFrame(cost_data)
                                cost_df.to_excel(writer, sheet_name='Cost Summary', index=False)
                            
                            # Audit results
                            if audit.untagged_resources:
                                untagged_df = pd.DataFrame(audit.untagged_resources)
                                untagged_df.to_excel(writer, sheet_name='Untagged Resources', index=False)
                            
                            if audit.stopped_instances:
                                stopped_df = pd.DataFrame(audit.stopped_instances)
                                stopped_df.to_excel(writer, sheet_name='Stopped Instances', index=False)
                            
                            if audit.unused_volumes:
                                volumes_df = pd.DataFrame(audit.unused_volumes)
                                volumes_df.to_excel(writer, sheet_name='Unused Volumes', index=False)
                        
                        excel_data = excel_buffer.getvalue()
                        
                        # Create download button for Excel
                        b64_excel = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{filename}" style="text-decoration: none;">'
                        href += '<button style="background-color:#28a745;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">'
                        href += '📊 Download Excel Report</button></a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("✅ Excel report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating Excel report: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one report section")
    
    with col3:
        if st.button("📋 Generate JSON Report"):
            if report_types:
                with st.spinner("Generating JSON report..."):
                    try:
                        # Create comprehensive JSON report
                        report_data = {
                            "report_metadata": {
                                "generated_at": datetime.now().isoformat(),
                                "cloud_providers": selected_clouds,
                                "report_scope": report_scope,
                                "report_types": report_types
                            },
                            "cost_summary": {
                                "total_cost": total_cost,
                                "provider_breakdown": provider_costs
                            },
                            "audit_results": {
                                "untagged_resources_count": len(audit.untagged_resources),
                                "stopped_instances_count": len(audit.stopped_instances),
                                "unused_volumes_count": len(audit.unused_volumes),
                                "untagged_resources": audit.untagged_resources[:20],  # Limit to 20 for size
                                "stopped_instances": audit.stopped_instances[:20],
                                "unused_volumes": audit.unused_volumes[:20]
                            },
                            "optimization_opportunities": {
                                "potential_monthly_savings": len(audit.stopped_instances) * 50 + len(audit.unused_volumes) * 25,
                                "recommendations": [
                                    "Tag untagged resources for better cost allocation",
                                    "Review and terminate unnecessary stopped instances",
                                    "Clean up unused storage volumes",
                                    "Implement automated cost monitoring"
                                ]
                            }
                        }
                        
                        # Convert to JSON
                        json_data = json.dumps(report_data, indent=2, default=str)
                        
                        # Create download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"finops_json_report_{timestamp}.json"
                        
                        b64_json = base64.b64encode(json_data.encode()).decode()
                        href = f'<a href="data:application/json;base64,{b64_json}" download="{filename}" style="text-decoration: none;">'
                        href += '<button style="background-color:#6f42c1;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">'
                        href += '📋 Download JSON Report</button></a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("✅ JSON report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating JSON report: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one report section")
    
    st.divider()
    
    # Report analytics and insights
    st.subheader("📈 Report Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💰 Cost Insights**")
        
        # Cost trend analysis
        if total_cost > 0:
            # Calculate some basic insights
            highest_cost_provider = max(provider_costs.items(), key=lambda x: x[1]) if provider_costs else ("N/A", 0)
            
            insights = [
                f"• Highest cost provider: {highest_cost_provider[0]} (${highest_cost_provider[1]:,.2f})",
                f"• Average cost per provider: ${total_cost / len([c for c in provider_costs.values() if c > 0]):,.2f}" if any(provider_costs.values()) else "• No cost data available",
                f"• Potential cost optimization: {((len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes)) / max(1, len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes) + 10)) * 100:.1f}% of resources need attention"
            ]
            
            for insight in insights:
                st.write(insight)
        else:
            st.info("No cost data available for analysis")
    
    with col2:
        st.write("**🎯 Optimization Opportunities**")
        
        optimization_score = max(0, 100 - (len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes)) * 5)
        
        st.metric("Optimization Score", f"{optimization_score:.0f}/100")
        
        if optimization_score >= 80:
            st.success("🟢 Excellent - Your infrastructure is well optimized!")
        elif optimization_score >= 60:
            st.warning("🟡 Good - Some optimization opportunities available")
        else:
            st.error("🔴 Needs attention - Multiple optimization opportunities identified")
        
        # Quick action items
        action_items = []
        if len(audit.untagged_resources) > 0:
            action_items.append(f"Tag {len(audit.untagged_resources)} untagged resources")
        if len(audit.stopped_instances) > 0:
            action_items.append(f"Review {len(audit.stopped_instances)} stopped instances")
        if len(audit.unused_volumes) > 0:
            action_items.append(f"Clean up {len(audit.unused_volumes)} unused volumes")
        
        if action_items:
            st.write("**Priority Actions:**")
            for item in action_items[:3]:
                st.write(f"• {item}")
        else:
            st.success("✅ No immediate actions required!")

def create_cost_visualizations(data_manager, selected_clouds):
    """Create enhanced cost visualizations"""
    st.subheader("📊 Advanced Cost Analytics")
    
    # Display enhanced cost metrics
    display_cost_metrics(data_manager, selected_clouds)
    
    # Get cost data
    provider_costs = data_manager.get_cost_summary_by_provider()
    if selected_clouds and "All Clouds" not in selected_clouds:
        provider_costs = {k: v for k, v in provider_costs.items() if k in selected_clouds}
    
    if not any(provider_costs.values()):
        st.info("No cost data available for the selected cloud providers.")
        return
    
    # Service breakdown visualization
    st.subheader("🔧 Service Cost Breakdown")
    service_costs = data_manager.get_cost_summary_by_service()
    
    if service_costs:
        # Top services chart
        top_services = dict(sorted(service_costs.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig_services = px.bar(
            x=list(top_services.keys()),
            y=list(top_services.values()),
            title="Top 10 Services by Cost",
            labels={'x': 'Service', 'y': 'Cost (USD)'},
            color=list(top_services.values()),
            color_continuous_scale='Blues'
        )
        fig_services.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_services, use_container_width=True)
    
    # Monthly trend analysis
    st.subheader("📈 Cost Trend Analysis")
    trends_df = data_manager.get_cost_trends(days=90)
    
    if not trends_df.empty:
        if selected_clouds and "All Clouds" not in selected_clouds:
            trends_df = trends_df[trends_df['provider'].isin(selected_clouds)]
        
        if not trends_df.empty:
            fig_trends = px.line(
                trends_df,
                x='date',
                y='cost_usd',
                color='provider',
                title="90-Day Cost Trends",
                markers=True
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No trend data available for selected providers.")
    else:
        st.info("No trend data available.")

def display_cost_metrics(data_manager, selected_clouds):
    """Display cost metrics with proper formatting."""
    try:
        # Get cost summary by provider
        provider_costs = data_manager.get_cost_summary_by_provider()
        
        # Calculate total cost for selected clouds
        total_cost = sum(provider_costs.get(cloud, 0) for cloud in selected_clouds)
        
        # Display total cost
        st.metric("💰 Total Cost", f"${total_cost:,.2f}")
        
        # Display cost breakdown for each selected provider
        for cloud in selected_clouds:
            if cloud in provider_costs:
                cost = provider_costs[cloud]
                if cost > 0:
                    st.subheader(f"{cloud} Cost Breakdown")
                    breakdown = data_manager.get_provider_cost_breakdown(cloud)
                    
                    # Display total cost for the provider
                    st.metric(f"Total {cloud} Cost", f"${cost:,.2f}")
                    
                    # Display service breakdown
                    if breakdown['service_breakdown']:
                        st.write("Top Services by Cost:")
                        for service, service_cost in breakdown['service_breakdown'].items():
                            st.write(f"- {service}: ${service_cost:,.2f}")
                    else:
                        st.write("No detailed cost breakdown available.")
                    
                    st.divider()
    except Exception as e:
        st.error(f"Error displaying cost metrics: {str(e)}")

def generate_pdf_report(data_manager, selected_clouds, report_types):
    """Generate comprehensive PDF report with charts and analysis"""
    try:
        # Create a BytesIO buffer for the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceBefore=20,
            spaceAfter=10
        )
        
        # Title and header
        story.append(Paragraph("FinOps Multi-Cloud Cost Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        clouds_text = ", ".join(selected_clouds) if "All Clouds" not in selected_clouds else "All Cloud Providers"
        
        metadata_data = [
            ["Report Generated:", report_date],
            ["Cloud Providers:", clouds_text],
            ["Report Types:", ", ".join(report_types)],
            ["Analysis Period:", "Last 90 days"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        if "executive_summary" in report_types:
            story.append(Paragraph("Executive Summary", heading_style))
            
            # Get key metrics
            provider_costs = data_manager.get_cost_summary_by_provider()
            total_cost = sum(provider_costs.get(cloud, 0) for cloud in selected_clouds)
            audit = data_manager.perform_finops_audit()
            
            # Calculate potential savings
            stopped_instances_savings = len(audit.stopped_instances) * 50  # Estimated $50/month per stopped instance
            unused_volumes_savings = len(audit.unused_volumes) * 25  # Estimated $25/month per unused volume
            total_potential_savings = stopped_instances_savings + unused_volumes_savings
            
            executive_summary = f"""
            This report provides a comprehensive analysis of your multi-cloud infrastructure costs and optimization opportunities.
            
            Key Findings:
            • Total monthly cost across selected providers: ${total_cost:,.2f}
            • Number of optimization opportunities identified: {len(audit.untagged_resources) + len(audit.stopped_instances) + len(audit.unused_volumes)}
            • Potential monthly savings: ${total_potential_savings:,.2f}
            • Untagged resources requiring attention: {len(audit.untagged_resources)}
            • Stopped instances consuming costs: {len(audit.stopped_instances)}
            • Unused volumes: {len(audit.unused_volumes)}
            
            Recommendations:
            • Implement proper resource tagging for better cost allocation
            • Review and terminate unnecessary stopped instances
            • Clean up unused storage volumes
            • Set up automated cost monitoring and alerting
            """
            
            story.append(Paragraph(executive_summary, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Cost Breakdown
        if "cost_analysis" in report_types:
            story.append(Paragraph("Cost Analysis by Provider", heading_style))
            
            cost_data = []
            cost_data.append(["Cloud Provider", "Monthly Cost", "Percentage", "Top Service"])
            
            for cloud in selected_clouds:
                if cloud in provider_costs:
                    cost = provider_costs[cloud]
                    percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                    breakdown = data_manager.get_provider_cost_breakdown(cloud)
                    top_service = max(breakdown['service_breakdown'].items(), key=lambda x: x[1])[0] if breakdown['service_breakdown'] else "N/A"
                    
                    cost_data.append([
                        cloud,
                        f"${cost:,.2f}",
                        f"{percentage:.1f}%",
                        top_service
                    ])
            
            cost_table = Table(cost_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(cost_table)
            story.append(Spacer(1, 20))
        
        # Resource Audit
        if "resource_audit" in report_types:
            story.append(Paragraph("Resource Audit & Optimization Opportunities", heading_style))
            
            # Untagged Resources
            if audit.untagged_resources:
                story.append(Paragraph("Untagged Resources (Top 10)", styles['Heading3']))
                untagged_data = [["Resource ID", "Service", "Provider", "Monthly Cost"]]
                
                for resource in audit.untagged_resources[:10]:
                    untagged_data.append([
                        resource.get('resource_id', 'N/A'),
                        resource.get('service', 'N/A'),
                        resource.get('provider', 'N/A'),
                        f"${resource.get('cost', 0):,.2f}"
                    ])
                
                untagged_table = Table(untagged_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                untagged_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(untagged_table)
                story.append(Spacer(1, 15))
            
            # Stopped Instances
            if audit.stopped_instances:
                story.append(Paragraph("Stopped Instances", styles['Heading3']))
                stopped_data = [["Instance ID", "Type", "Provider", "Estimated Monthly Cost"]]
                
                for instance in audit.stopped_instances[:10]:
                    stopped_data.append([
                        instance.get('instance_id', 'N/A'),
                        instance.get('instance_type', 'N/A'),
                        instance.get('provider', 'N/A'),
                        f"${instance.get('estimated_cost', 50):,.2f}"
                    ])
                
                stopped_table = Table(stopped_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                stopped_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(stopped_table)
                story.append(Spacer(1, 15))
            
            # Unused Volumes
            if audit.unused_volumes:
                story.append(Paragraph("Unused Storage Volumes", styles['Heading3']))
                volumes_data = [["Volume ID", "Size (GB)", "Provider", "Monthly Cost"]]
                
                for volume in audit.unused_volumes[:10]:
                    volumes_data.append([
                        volume.get('volume_id', 'N/A'),
                        str(volume.get('size_gb', 'N/A')),
                        volume.get('provider', 'N/A'),
                        f"${volume.get('monthly_cost', 25):,.2f}"
                    ])
                
                volumes_table = Table(volumes_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                volumes_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(volumes_table)
                story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Actionable Recommendations", heading_style))
        
        recommendations = [
            "1. Implement comprehensive resource tagging strategy for better cost allocation and governance",
            "2. Set up automated monitoring for stopped instances and unused resources",
            "3. Establish monthly cost review meetings with stakeholders",
            "4. Consider reserved instances for predictable workloads to reduce costs",
            "5. Implement auto-scaling policies to optimize resource utilization",
            "6. Set up budget alerts and spending limits for proactive cost management",
            "7. Regular audits of storage classes and cleanup of old data",
            "8. Consider multi-cloud cost optimization tools for better visibility"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 8))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

def create_download_button(pdf_data, filename):
    """Create a download button for the PDF report"""
    if pdf_data:
        b64_pdf = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" style="text-decoration: none;">'
        href += '<button style="background-color:#0066cc;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">'
        href += '📄 Download PDF Report</button></a>'
        return href
    return None

def main():
    # Initialize session state
    init_session_state()
    
    # Initialize components
    client = get_openai_client()
    data_manager = load_enhanced_data()
    
    # Header
    st.title("☁️ Enhanced FinOps AI Agent")
    st.markdown("**Multi-Cloud Cost Management & Optimization Platform**")
    st.markdown("*Advanced FinOps capabilities for unified cloud cost management, audit reporting, and automated alerting*")
    
    # Cloud provider selection
    selected_clouds, analysis_scope = render_cloud_selector()
    
    # Check and display alerts
    check_and_display_alerts(data_manager, selected_clouds)
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.title("🎛️ FinOps Control Center")
        
        # Quick refresh
        if st.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Spending limits configuration
        with st.expander("💰 Configure Spending Limits"):
            render_spending_limits_config()
        
        # Quick stats for selected clouds
        st.subheader("📊 Quick Overview")
        provider_costs = data_manager.get_cost_summary_by_provider()
        if selected_clouds and "All Clouds" not in selected_clouds:
            provider_costs = {k: v for k, v in provider_costs.items() if k in selected_clouds}
        
        total_cost = sum(provider_costs.values()) if provider_costs else 0
        st.metric("💰 Total Cost", f"${total_cost:,.2f}")
        st.metric("☁️ Selected Clouds", len(selected_clouds) if "All Clouds" not in selected_clouds else "All")
        
        # Feature highlights
        st.subheader("✨ Enhanced Features")
        st.success("☁️ Multi-Cloud Selection")
        st.success("🧠 AI Analytics & Predictions")
        st.success("📊 Unified Dashboard")
        st.success("🔔 Smart Alerts & Anomalies")
        st.success("🎯 ML Optimization Engine")
        st.success("🏷️ Intelligent Tagging")
        st.success("📈 6-Month Trends")
        st.success("🔍 Comprehensive Audit")
        st.success("📄 Multi-Format Reports")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💬 AI Assistant", 
        "🧠 AI Analytics",
        "📊 Unified Dashboard", 
        "📈 Cost Trends",
        "🔍 Cloud Audit", 
        "📊 Report Generation",
        "🎛️ Settings"
    ])
    
    with tab1:
        # Enhanced AI Assistant
        st.subheader("🤖 AI-Powered FinOps Assistant")
        
        # Initialize session state for messages
        if 'enhanced_messages' not in st.session_state:
            system_message = create_enhanced_system_message(data_manager, selected_clouds)
            st.session_state.enhanced_messages = [
                {"role": "system", "content": system_message}
            ]
        
        # Display conversation
        for msg in st.session_state.enhanced_messages:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            elif msg['role'] == 'assistant':
                with st.chat_message("assistant"):
                    st.write(msg['content'])
        
        # Suggested queries based on selected clouds
        st.subheader("🚀 Quick Actions")
        
        # Interactive Commands Section
        st.write("**🎮 Interactive Commands:**")
        interactive_commands = [
            "Show untagged resources",
            "Show stopped instances", 
            "Show unused volumes",
            "Show project costs",
            "Detect cost anomalies",
            "Show optimization recommendations",
            "Suggest tags for untagged resources",
            "Show priority alerts"
        ]
        
        cols = st.columns(4)
        for i, command in enumerate(interactive_commands):
            if cols[i % 4].button(f"🔧 {command}", key=f"cmd_{i}"):
                st.session_state.enhanced_messages.append({"role": "user", "content": command})
                st.rerun()
        
        # AI Analysis Queries
        st.write("**🧠 AI Analysis Queries:**")
        if selected_clouds and "All Clouds" not in selected_clouds:
            suggested_queries = [
                f"Predict costs for {', '.join(selected_clouds)} next 3 months",
                f"Detect anomalies in {', '.join(selected_clouds)} spending",
                f"Show optimization recommendations for {', '.join(selected_clouds)}",
                f"Generate smart tags for untagged resources",
                f"Show priority alerts for {', '.join(selected_clouds)}",
                f"Analyze rightsizing opportunities in {', '.join(selected_clouds)}"
            ]
        else:
            suggested_queries = [
                "Predict costs for next 3 months",
                "Detect cost anomalies across all clouds",
                "Show optimization recommendations",
                "Suggest tags for untagged resources", 
                "Show priority alerts",
                "Generate cost forecasts with confidence intervals"
            ]
        
        cols = st.columns(2)
        for i, query in enumerate(suggested_queries):
            if cols[i % 2].button(query, key=f"enhanced_quick_{i}"):
                st.session_state.enhanced_messages.append({"role": "user", "content": query})
                st.rerun()
        
        # Command Examples Section
        with st.expander("📋 Interactive Command Examples"):
            st.write("**🎯 AI Analytics Commands:**")
            st.code("Predict costs for next 3 months")
            st.code("Detect cost anomalies")
            st.code("Show optimization recommendations")
            st.code("Suggest tags for untagged resources")
            st.code("Show priority alerts")
            
            st.write("**🏷️ Tagging Commands:**")
            st.code("Tag resource ec2-25 with Project=WebApp,Environment=Production")
            st.code("Tag all untagged EC2 resources with Project=WebApp,Team=DevOps")
            
            st.write("**⚙️ Instance Management:**")
            st.code("Terminate stopped instance i-009d67ca44d9")
            st.code("Start stopped instance i-002c9554464b")
            
            st.write("**💾 Volume Management:**")
            st.code("Delete unused volume vol-123456789abcdef")
            st.code("Attach volume vol-123456789abcdef to instance i-009d67ca44d9")
            
            st.write("**📊 Project Analysis:**")
            st.code("Show costs for project WebApp")
            st.code("Show costs for project Analytics")
        
        # Chat input
        if prompt := st.chat_input("Ask about your multi-cloud costs and optimization..."):
            st.session_state.enhanced_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing your multi-cloud environment..."):
                    try:
                        # First, check if this is an interactive command
                        command_response = data_manager.process_chat_command(prompt)
                        
                        if command_response.success:
                            # Handle successful command execution
                            st.success(f"✅ {command_response.message}")
                            
                            # Display additional data if available
                            if command_response.data:
                                if 'untagged_count' in command_response.data:
                                    st.info(f"Found {command_response.data['untagged_count']} untagged resources with ${command_response.data['total_cost']:.2f} in costs")
                                
                                elif 'stopped_instances' in command_response.data:
                                    if command_response.data['stopped_instances']:
                                        st.dataframe(pd.DataFrame(command_response.data['stopped_instances']))
                                
                                elif 'unused_volumes' in command_response.data:
                                    if command_response.data['unused_volumes']:
                                        st.dataframe(pd.DataFrame(command_response.data['unused_volumes']))
                                
                                elif 'project_costs' in command_response.data:
                                    project_data = command_response.data['project_costs']
                                    project_df = pd.DataFrame([
                                        {
                                            'Project': name, 
                                            'Total Cost': f"${data['total_cost']:.2f}",
                                            'Resources': data['resource_count']
                                        }
                                        for name, data in project_data.items()
                                    ])
                                    st.dataframe(project_df)
                            
                            # Add command result to conversation context
                            assistant_reply = f"Command executed successfully: {command_response.message}"
                            
                        else:
                            # Not a command or command failed, use AI assistant
                            if "didn't understand that command" not in command_response.message:
                                st.warning(f"⚠️ {command_response.message}")
                            
                            # Update system message with current context
                            updated_system_message = create_enhanced_system_message(data_manager, selected_clouds)
                            st.session_state.enhanced_messages[0] = {"role": "system", "content": updated_system_message}
                            
                            response = client.chat.completions.create(
                                messages=st.session_state.enhanced_messages,
                                max_tokens=Config.MAX_TOKENS,
                                temperature=Config.TEMPERATURE,
                                top_p=Config.TOP_P,
                                model=Config.AZURE_DEPLOYMENT
                            )
                            assistant_reply = response.choices[0].message.content
                            st.write(assistant_reply)
                        
                        st.session_state.enhanced_messages.append({"role": "assistant", "content": assistant_reply})
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Clear chat
        if st.button("🗑️ Clear Conversation"):
            system_message = create_enhanced_system_message(data_manager, selected_clouds)
            st.session_state.enhanced_messages = [{"role": "system", "content": system_message}]
            st.rerun()
    
    with tab2:
        # AI Analytics Tab - NEW ENHANCED FEATURES
        st.subheader("🧠 AI-Powered Analytics & Insights")
        
        # Create columns for different AI features
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost Prediction Section
            st.markdown("### 📈 Cost Predictions")
            with st.expander("🔮 Forecast Future Costs", expanded=True):
                pred_provider = st.selectbox(
                    "Select Provider for Prediction", 
                    ["All Providers"] + (selected_clouds if "All Clouds" not in selected_clouds else ["AWS", "Azure", "GCP"])
                )
                months_ahead = st.slider("Months to Predict", 1, 6, 3)
                
                if st.button("🔮 Generate Predictions"):
                    provider_filter = None if pred_provider == "All Providers" else pred_provider
                    predictions = data_manager.predict_monthly_costs(provider_filter, months_ahead)
                    
                    if predictions and "insufficient_data" not in predictions:
                        st.success("✅ Predictions Generated Successfully!")
                        
                        # Create prediction visualization
                        pred_data = []
                        for month, pred in predictions.items():
                            month_num = int(month.split('_')[1])
                            pred_data.append({
                                'Month': f"Month +{month_num}",
                                'Predicted Cost': pred.predicted_cost,
                                'Lower Bound': pred.confidence_interval[0],
                                'Upper Bound': pred.confidence_interval[1],
                                'Trend': pred.trend
                            })
                        
                        pred_df = pd.DataFrame(pred_data)
                        
                        # Display chart
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df['Month'],
                            y=pred_df['Predicted Cost'],
                            mode='lines+markers',
                            name='Predicted Cost',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df['Month'],
                            y=pred_df['Upper Bound'],
                            fill=None,
                            mode='lines',
                            line_color='lightblue',
                            name='Upper Bound'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df['Month'],
                            y=pred_df['Lower Bound'],
                            fill='tonexty',
                            mode='lines',
                            line_color='lightblue',
                            name='Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title="💡 AI-Powered Cost Predictions",
                            xaxis_title="Time Period",
                            yaxis_title="Predicted Cost ($)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction details
                        for month, pred in predictions.items():
                            month_num = month.split('_')[1]
                            trend_emoji = "📈" if pred.trend == "increasing" else "📉" if pred.trend == "decreasing" else "➡️"
                            st.info(f"**Month {month_num}:** ${pred.predicted_cost:,.2f} {trend_emoji} ({pred.trend})")
                            st.write(f"💡 {pred.recommendation}")
                    else:
                        st.warning("⚠️ Insufficient historical data for accurate predictions. Need at least 3 months of data.")
            
            # Smart Tagging Section
            st.markdown("### 🏷️ Intelligent Tagging")
            with st.expander("🤖 AI Tag Suggestions", expanded=False):
                if st.button("🧠 Generate Smart Tags"):
                    tag_suggestions = data_manager.suggest_intelligent_tags()
                    
                    if tag_suggestions:
                        st.success(f"✅ Found {len(tag_suggestions)} tag suggestions!")
                        
                        # High confidence suggestions
                        high_conf = [s for s in tag_suggestions if s.confidence_score >= 0.7]
                        if high_conf:
                            st.markdown("**🎯 High Confidence Suggestions:**")
                            for suggestion in high_conf[:5]:
                                with st.container():
                                    st.write(f"**{suggestion.resource_id}** (Confidence: {suggestion.confidence_score:.0%})")
                                    tags_str = ", ".join([f"{k}={v}" for k, v in suggestion.suggested_tags.items()])
                                    st.code(f"Suggested tags: {tags_str}")
                                    st.caption(f"Reasoning: {suggestion.reasoning}")
                                    
                                    # Quick apply button
                                    if st.button(f"Apply Tags", key=f"apply_{suggestion.resource_id}"):
                                        tag_command = f"Tag resource {suggestion.resource_id} with {tags_str}"
                                        result = data_manager.process_chat_command(tag_command)
                                        if result.success:
                                            st.success("✅ Tags applied successfully!")
                                        else:
                                            st.error(f"❌ Failed to apply tags: {result.message}")
                                    st.divider()
                    else:
                        st.info("✅ All resources appear to be properly tagged!")
        
        with col2:
            # Anomaly Detection Section
            st.markdown("### 🚨 Anomaly Detection")
            with st.expander("🔍 Detect Cost Anomalies", expanded=True):
                sensitivity = st.selectbox("Detection Sensitivity", ["high", "medium", "low"], index=1)
                
                if st.button("🔍 Detect Anomalies"):
                    anomalies = data_manager.detect_cost_anomalies(sensitivity)
                    
                    if anomalies:
                        st.warning(f"⚠️ Found {len(anomalies)} cost anomalies!")
                        
                        for i, anomaly in enumerate(anomalies[:5], 1):
                            severity_colors = {
                                "critical": "🔴",
                                "high": "🟠", 
                                "medium": "🟡",
                                "low": "🟢"
                            }
                            
                            with st.container():
                                st.markdown(f"**{severity_colors.get(anomaly.severity, '🔵')} Anomaly {i}: {anomaly.resource_id}**")
                                st.write(f"**Issue:** {anomaly.description}")
                                st.write(f"**Cost Impact:** ${anomaly.cost_impact:.2f}")
                                st.write(f"**Recommended Action:** {anomaly.recommended_action}")
                                st.divider()
                    else:
                        st.success("✅ No anomalies detected. Your spending patterns look normal!")
            
            # Optimization Recommendations Section
            st.markdown("### 🎯 Optimization Recommendations")
            with st.expander("💡 AI Optimization Engine", expanded=False):
                opt_focus = st.selectbox(
                    "Focus Area", 
                    ["all", "rightsizing", "storage", "scheduling", "purchasing"]
                )
                
                if st.button("🎯 Generate Recommendations"):
                    recommendations = data_manager.generate_optimization_recommendations(opt_focus)
                    
                    if recommendations:
                        total_savings = sum(r.potential_savings for r in recommendations)
                        st.success(f"💰 Found ${total_savings:.2f}/month in potential savings!")
                        
                        for i, rec in enumerate(recommendations[:3], 1):
                            with st.container():
                                effort_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                                st.markdown(f"**💡 Recommendation {i}: {rec.category.title()}**")
                                st.write(f"**Resource:** {rec.resource_type}")
                                st.write(f"**Current Cost:** ${rec.current_cost:.2f}/month")
                                st.write(f"**Potential Savings:** ${rec.potential_savings:.2f}/month")
                                st.write(f"**Confidence:** {rec.confidence_score:.0%}")
                                st.write(f"**Implementation Effort:** {effort_colors.get(rec.implementation_effort, '🔵')} {rec.implementation_effort.title()}")
                                st.write(f"**Description:** {rec.description}")
                                
                                if rec.action_steps:
                                    st.write("**Action Steps:**")
                                    for step in rec.action_steps:
                                        st.write(f"• {step}")
                                st.divider()
                    else:
                        st.info("✅ No immediate optimization opportunities found!")
        
        # Intelligent Alerts Section
        st.markdown("### 🚨 Intelligent Alerts Dashboard")
        with st.expander("🔔 Priority Alerts & Notifications", expanded=True):
            alert_priority = st.selectbox("Filter by Priority", ["all", "critical", "high", "medium"])
            
            alerts = data_manager.get_intelligent_alerts(alert_priority)
            
            if alerts:
                st.warning(f"⚠️ {len(alerts)} alerts require attention!")
                
                # Create alert metrics
                alert_counts = {}
                total_cost_impact = 0
                
                for alert in alerts:
                    priority = alert['priority']
                    alert_counts[priority] = alert_counts.get(priority, 0) + 1
                    total_cost_impact += abs(alert.get('cost_impact', 0))
                
                cols = st.columns(len(alert_counts) + 1)
                for i, (priority, count) in enumerate(alert_counts.items()):
                    priority_emojis = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    cols[i].metric(f"{priority_emojis.get(priority, '🔵')} {priority.title()}", count)
                
                cols[-1].metric("💰 Total Impact", f"${total_cost_impact:.2f}")
                
                # Display alerts
                for alert in alerts[:5]:
                    priority_colors = {
                        "critical": "error",
                        "high": "warning", 
                        "medium": "info",
                        "low": "success"
                    }
                    
                    with st.container():
                        getattr(st, priority_colors.get(alert['priority'], 'info'))(
                            f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}"
                        )
            else:
                st.success("✅ No priority alerts at this time. Everything looks good!")

    with tab3:
        render_unified_dashboard(data_manager, selected_clouds)
        create_cost_visualizations(data_manager, selected_clouds)
    
    with tab4:
        render_six_month_trends(data_manager, selected_clouds)
    
    with tab5:
        render_cloud_audit_report(data_manager, selected_clouds)
    
    with tab6:
        render_comprehensive_reporting(data_manager, selected_clouds)
    
    with tab7:
        st.subheader("🎛️ Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alert Configuration**")
            st.write("Configure when and how you receive notifications about your cloud spending and resource optimization opportunities.")
            
        with col2:
            st.write("**Export Settings**")
            default_export_dir = st.text_input("Default Export Directory", value="./reports")
            auto_generate_reports = st.checkbox("Auto-generate monthly reports", value=False)
            include_forecasting = st.checkbox("Include cost forecasting in reports", value=True)

if __name__ == "__main__":
    main() 