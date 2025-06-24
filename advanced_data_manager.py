import os
import json
import pandas as pd
import random
import datetime
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from config import Config
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TimeRange(Enum):
    SEVEN_DAYS = 7
    THIRTY_DAYS = 30
    NINETY_DAYS = 90
    SIX_MONTHS = 180
    CURRENT_MONTH = "current_month"
    PREVIOUS_MONTH = "previous_month"

@dataclass
class BudgetInfo:
    name: str
    limit: float
    actual_spend: float
    percentage_used: float
    status: str  # "on_track", "warning", "exceeded"

@dataclass
class ResourceAudit:
    untagged_resources: List[Dict]
    stopped_instances: List[Dict]
    unused_volumes: List[Dict]
    unused_eips: List[Dict]
    budget_alerts: List[BudgetInfo]

@dataclass
class ChatResponse:
    success: bool
    message: str
    data: Optional[Dict] = None

@dataclass
class CostPrediction:
    """AI-powered cost prediction results"""
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    trend: str  # "increasing", "decreasing", "stable"
    factors: List[str]
    recommendation: str

@dataclass
class AnomalyAlert:
    """Anomaly detection results"""
    resource_id: str
    anomaly_type: str  # "cost_spike", "unusual_pattern", "sudden_stop"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommended_action: str
    cost_impact: float

@dataclass
class OptimizationRecommendation:
    """AI-generated optimization recommendations"""
    category: str  # "rightsizing", "scheduling", "storage", "purchasing"
    resource_type: str
    current_cost: float
    potential_savings: float
    confidence_score: float
    implementation_effort: str  # "low", "medium", "high"
    description: str
    action_steps: List[str]

@dataclass
class SmartTag:
    """AI-suggested resource tags"""
    resource_id: str
    suggested_tags: Dict[str, str]
    confidence_score: float
    reasoning: str

class AdvancedDataManager:
    def __init__(self):
        self.cost_data_df = None
        self.dummy_data = None
        self.budgets = []
        self.ec2_instances = []
        self.volumes = []  # Add volumes tracking
        # AI Enhancement: Add ML model storage
        self.cost_predictions = {}
        self.anomaly_history = []
        self.optimization_cache = {}
        self.load_data()
        self.generate_sample_budgets()
        self.generate_sample_ec2_data()
        self.generate_sample_volumes()  # Add volumes generation
    
    # ============= AI ENHANCEMENT: PREDICTIVE ANALYTICS =============
    
    def predict_monthly_costs(self, provider: Optional[str] = None, months_ahead: int = 3) -> Dict[str, CostPrediction]:
        """AI-powered cost forecasting with confidence intervals"""
        if self.cost_data_df.empty:
            return {}
        
        predictions = {}
        
        # Filter by provider if specified
        df = self.cost_data_df if not provider else self.cost_data_df[self.cost_data_df['provider'] == provider]
        
        # Group by month for trend analysis
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['date'].dt.to_period('M')
        monthly_costs = df_monthly.groupby('month')['cost_usd'].sum().sort_index()
        
        if len(monthly_costs) < 3:
            return {"insufficient_data": CostPrediction(0, (0, 0), "unknown", ["Insufficient historical data"], "Collect more historical data for accurate predictions")}
        
        # Simple linear regression for trend
        x = np.arange(len(monthly_costs))
        y = monthly_costs.values
        
        # Calculate trend
        slope = np.polyfit(x, y, 1)[0]
        
        # Generate predictions
        for i in range(1, months_ahead + 1):
            future_x = len(monthly_costs) + i - 1
            predicted_cost = np.polyval(np.polyfit(x, y, 1), future_x)
            
            # Calculate confidence interval (simple approach)
            std_dev = np.std(y)
            confidence_interval = (
                max(0, predicted_cost - 1.96 * std_dev),
                predicted_cost + 1.96 * std_dev
            )
            
            # Determine trend
            if slope > monthly_costs.mean() * 0.05:  # More than 5% growth
                trend = "increasing"
                factors = ["Historical growth trend", "Potential scaling of resources"]
            elif slope < -monthly_costs.mean() * 0.05:  # More than 5% decrease
                trend = "decreasing"
                factors = ["Cost optimization efforts", "Resource cleanup"]
            else:
                trend = "stable"
                factors = ["Consistent usage patterns", "Steady workload"]
            
            # Generate recommendation
            if trend == "increasing" and predicted_cost > monthly_costs.mean() * 1.3:
                recommendation = "🚨 High cost growth predicted. Consider implementing cost controls and optimization measures."
            elif trend == "decreasing":
                recommendation = "✅ Positive cost trend. Continue current optimization efforts."
            else:
                recommendation = "📊 Stable costs predicted. Monitor for any unusual changes."
            
            predictions[f"month_{i}"] = CostPrediction(
                predicted_cost=round(predicted_cost, 2),
                confidence_interval=(round(confidence_interval[0], 2), round(confidence_interval[1], 2)),
                trend=trend,
                factors=factors,
                recommendation=recommendation
            )
        
        return predictions
    
    def detect_cost_anomalies(self, sensitivity: str = "medium") -> List[AnomalyAlert]:
        """AI-powered anomaly detection for cost patterns"""
        if self.cost_data_df.empty:
            return []
        
        anomalies = []
        sensitivity_thresholds = {
            "low": 3.0,      # 3 standard deviations
            "medium": 2.5,   # 2.5 standard deviations
            "high": 2.0      # 2 standard deviations
        }
        threshold = sensitivity_thresholds.get(sensitivity, 2.5)
        
        # Analyze by resource for cost spikes
        for resource in self.cost_data_df['resource'].unique():
            resource_data = self.cost_data_df[self.cost_data_df['resource'] == resource]
            if len(resource_data) < 3:
                continue
                
            costs = resource_data['cost_usd'].values
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            if std_cost == 0:  # No variance
                continue
            
            # Check for anomalies
            for idx, cost in enumerate(costs):
                z_score = abs(cost - mean_cost) / std_cost
                
                if z_score > threshold:
                    severity = "critical" if z_score > 4 else "high" if z_score > 3 else "medium"
                    
                    anomalies.append(AnomalyAlert(
                        resource_id=resource,
                        anomaly_type="cost_spike" if cost > mean_cost else "unusual_low_cost",
                        severity=severity,
                        description=f"Cost anomaly detected: ${cost:.2f} vs average ${mean_cost:.2f} (Z-score: {z_score:.2f})",
                        recommended_action="Investigate resource usage and configuration changes",
                        cost_impact=abs(cost - mean_cost)
                    ))
        
        # Check for stopped resources still incurring costs
        for instance in self.ec2_instances:
            if instance['state'] == 'stopped' and instance['monthly_cost'] > 10:
                anomalies.append(AnomalyAlert(
                    resource_id=instance['instance_id'],
                    anomaly_type="stopped_instance_cost",
                    severity="medium",
                    description=f"Stopped instance still incurring ${instance['monthly_cost']:.2f}/month",
                    recommended_action="Consider terminating if no longer needed",
                    cost_impact=instance['monthly_cost']
                ))
        
        return sorted(anomalies, key=lambda x: x.cost_impact, reverse=True)
    
    # ============= AI ENHANCEMENT: INTELLIGENT OPTIMIZATION =============
    
    def generate_optimization_recommendations(self, focus_area: str = "all") -> List[OptimizationRecommendation]:
        """AI-powered optimization recommendations"""
        recommendations = []
        
        if self.cost_data_df.empty:
            return recommendations
        
        # Right-sizing recommendations
        if focus_area in ["all", "rightsizing"]:
            recommendations.extend(self._analyze_rightsizing_opportunities())
        
        # Storage optimization
        if focus_area in ["all", "storage"]:
            recommendations.extend(self._analyze_storage_optimization())
        
        # Scheduling opportunities
        if focus_area in ["all", "scheduling"]:
            recommendations.extend(self._analyze_scheduling_opportunities())
        
        # Purchasing optimization (Reserved Instances, Savings Plans)
        if focus_area in ["all", "purchasing"]:
            recommendations.extend(self._analyze_purchasing_optimization())
        
        return sorted(recommendations, key=lambda x: x.potential_savings, reverse=True)
    
    def _analyze_rightsizing_opportunities(self) -> List[OptimizationRecommendation]:
        """Analyze right-sizing opportunities using AI patterns"""
        recommendations = []
        
        # Analyze EC2 instances for right-sizing
        for instance in self.ec2_instances:
            if instance['state'] == 'running':
                monthly_cost = instance['monthly_cost']
                instance_type = instance['instance_type']
                
                # Simple heuristic: if cost is high and it's a large instance type
                if monthly_cost > 200 and any(size in instance_type for size in ['large', 'xlarge', '2xlarge']):
                    potential_savings = monthly_cost * 0.3  # Assume 30% savings potential
                    
                    recommendations.append(OptimizationRecommendation(
                        category="rightsizing",
                        resource_type="EC2 Instance",
                        current_cost=monthly_cost,
                        potential_savings=potential_savings,
                        confidence_score=0.7,
                        implementation_effort="medium",
                        description=f"Instance {instance['instance_id']} ({instance_type}) may be oversized",
                        action_steps=[
                            "Monitor CPU and memory utilization",
                            "Analyze usage patterns over 2-4 weeks",
                            "Consider smaller instance type if utilization < 40%",
                            "Test with smaller instance in non-production first"
                        ]
                    ))
        
        return recommendations
    
    def _analyze_storage_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze storage optimization opportunities"""
        recommendations = []
        
        # Analyze storage services
        storage_services = ['S3', 'BlobStorage', 'CloudStorage', 'EBS', 'Disk']
        storage_data = self.cost_data_df[self.cost_data_df['service'].isin(storage_services)]
        
        for _, row in storage_data.iterrows():
            if row['cost_usd'] > 50:  # Focus on significant storage costs
                potential_savings = row['cost_usd'] * 0.4  # Assume 40% savings with lifecycle policies
                
                recommendations.append(OptimizationRecommendation(
                    category="storage",
                    resource_type=row['service'],
                    current_cost=row['cost_usd'],
                    potential_savings=potential_savings,
                    confidence_score=0.8,
                    implementation_effort="low",
                    description=f"Storage cost optimization for {row['resource']}",
                    action_steps=[
                        "Implement intelligent tiering policies",
                        "Archive infrequently accessed data",
                        "Enable compression if not already active",
                        "Review and delete orphaned snapshots"
                    ]
                ))
        
        return recommendations
    
    def _analyze_scheduling_opportunities(self) -> List[OptimizationRecommendation]:
        """Analyze resource scheduling opportunities"""
        recommendations = []
        
        # Look for development/staging resources that could be scheduled
        for instance in self.ec2_instances:
            if instance['state'] == 'running' and 'tags' in instance and instance['tags']:
                if 'Development' in instance['tags'] or 'Staging' in instance['tags']:
                    monthly_cost = instance['monthly_cost']
                    # Assume 50% savings by running only during business hours
                    potential_savings = monthly_cost * 0.5
                    
                    recommendations.append(OptimizationRecommendation(
                        category="scheduling",
                        resource_type="EC2 Instance",
                        current_cost=monthly_cost,
                        potential_savings=potential_savings,
                        confidence_score=0.9,
                        implementation_effort="low",
                        description=f"Development/Staging instance {instance['instance_id']} running 24/7",
                        action_steps=[
                            "Implement automated start/stop scheduling",
                            "Run only during business hours (8AM-6PM)",
                            "Stop instances during weekends",
                            "Use AWS Instance Scheduler or similar tools"
                        ]
                    ))
        
        return recommendations
    
    def _analyze_purchasing_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze Reserved Instance and Savings Plan opportunities"""
        recommendations = []
        
        # Analyze running instances for RI opportunities
        running_instances = [inst for inst in self.ec2_instances if inst['state'] == 'running']
        
        if len(running_instances) >= 3:  # Need multiple instances to justify RIs
            total_on_demand_cost = sum(inst['monthly_cost'] for inst in running_instances)
            potential_savings = total_on_demand_cost * 0.3  # Assume 30% RI savings
            
            recommendations.append(OptimizationRecommendation(
                category="purchasing",
                resource_type="Reserved Instances",
                current_cost=total_on_demand_cost,
                potential_savings=potential_savings,
                confidence_score=0.8,
                implementation_effort="high",
                description="Multiple long-running instances suitable for Reserved Instances",
                action_steps=[
                    "Analyze 6-month usage patterns",
                    "Purchase 1-year term Reserved Instances for stable workloads",
                    "Consider Convertible RIs for flexibility",
                    "Use RI recommendations from cloud provider console"
                ]
            ))
        
        return recommendations
    
    # ============= AI ENHANCEMENT: INTELLIGENT TAGGING =============
    
    def suggest_intelligent_tags(self, untagged_only: bool = True) -> List[SmartTag]:
        """AI-powered resource tagging suggestions"""
        suggestions = []
        
        if self.cost_data_df.empty:
            return suggestions
        
        # Filter resources that need tagging
        if untagged_only:
            untagged_resources = self.cost_data_df[
                (self.cost_data_df['tags'].isna()) | (self.cost_data_df['tags'] == '')
            ]
        else:
            untagged_resources = self.cost_data_df
        
        for _, row in untagged_resources.iterrows():
            resource_name = row['resource']
            service = row['service']
            provider = row['provider']
            
            # AI logic to suggest tags based on patterns
            suggested_tags = {}
            reasoning_parts = []
            
            # Infer environment from resource name
            if any(env in resource_name.lower() for env in ['prod', 'production']):
                suggested_tags['Environment'] = 'Production'
                reasoning_parts.append("'prod' pattern in name")
            elif any(env in resource_name.lower() for env in ['dev', 'development']):
                suggested_tags['Environment'] = 'Development'
                reasoning_parts.append("'dev' pattern in name")
            elif any(env in resource_name.lower() for env in ['staging', 'stage']):
                suggested_tags['Environment'] = 'Staging'
                reasoning_parts.append("'staging' pattern in name")
            else:
                suggested_tags['Environment'] = 'Unknown'
                reasoning_parts.append("no clear environment pattern")
            
            # Infer team/project from service type and naming patterns
            if service in ['EC2', 'VM', 'ComputeEngine']:
                if any(pattern in resource_name.lower() for pattern in ['web', 'frontend', 'ui']):
                    suggested_tags['Team'] = 'Frontend'
                    reasoning_parts.append("web/frontend pattern detected")
                elif any(pattern in resource_name.lower() for pattern in ['api', 'backend', 'service']):
                    suggested_tags['Team'] = 'Backend'
                    reasoning_parts.append("API/backend pattern detected")
                elif any(pattern in resource_name.lower() for pattern in ['db', 'database', 'data']):
                    suggested_tags['Team'] = 'DataTeam'
                    reasoning_parts.append("database pattern detected")
                else:
                    suggested_tags['Team'] = 'DevOps'
                    reasoning_parts.append("default compute team assignment")
            
            # Infer cost center based on service criticality
            if suggested_tags.get('Environment') == 'Production':
                suggested_tags['CostCenter'] = 'Production-Ops'
            else:
                suggested_tags['CostCenter'] = 'Development'
            
            # Confidence score based on how many patterns we matched
            confidence_score = len(reasoning_parts) / 4.0  # Max 4 reasoning points
            
            suggestions.append(SmartTag(
                resource_id=resource_name,
                suggested_tags=suggested_tags,
                confidence_score=min(confidence_score, 1.0),
                reasoning=f"Based on: {', '.join(reasoning_parts)}"
            ))
        
        return sorted(suggestions, key=lambda x: x.confidence_score, reverse=True)
    
    # ============= AI ENHANCEMENT: ADVANCED ALERTING =============
    
    def get_intelligent_alerts(self, priority_filter: str = "all") -> List[Dict]:
        """Generate prioritized, intelligent alerts"""
        alerts = []
        
        # Cost anomaly alerts
        anomalies = self.detect_cost_anomalies()
        for anomaly in anomalies:
            if priority_filter == "all" or anomaly.severity in [priority_filter, "high", "critical"]:
                alerts.append({
                    "type": "anomaly",
                    "priority": anomaly.severity,
                    "title": f"Cost Anomaly Detected: {anomaly.resource_id}",
                    "message": anomaly.description,
                    "action": anomaly.recommended_action,
                    "cost_impact": anomaly.cost_impact,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Budget burn rate alerts
        for budget in self.budgets:
            if budget.percentage_used > 80:
                priority = "critical" if budget.percentage_used > 100 else "high" if budget.percentage_used > 90 else "medium"
                
                if priority_filter == "all" or priority in [priority_filter, "high", "critical"]:
                    alerts.append({
                        "type": "budget",
                        "priority": priority,
                        "title": f"Budget Alert: {budget.name}",
                        "message": f"Budget utilization at {budget.percentage_used:.1f}%",
                        "action": "Review spending and consider cost controls",
                        "cost_impact": budget.actual_spend - budget.limit if budget.percentage_used > 100 else 0,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
        
        # Optimization opportunity alerts
        recommendations = self.generate_optimization_recommendations()
        high_impact_recs = [r for r in recommendations if r.potential_savings > 100]
        
        for rec in high_impact_recs[:3]:  # Top 3 recommendations only
            if priority_filter == "all" or "medium" in [priority_filter]:
                alerts.append({
                    "type": "optimization",
                    "priority": "medium",
                    "title": f"Optimization Opportunity: {rec.category.title()}",
                    "message": rec.description,
                    "action": f"Potential savings: ${rec.potential_savings:.2f}/month",
                    "cost_impact": -rec.potential_savings,  # Negative because it's savings
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Sort by priority and cost impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda x: (priority_order.get(x["priority"], 3), -abs(x["cost_impact"])))
        
        return alerts
    
    # ============= AI ENHANCEMENT: ENHANCED CHAT COMMANDS =============
    
    def process_chat_command(self, user_message: str) -> ChatResponse:
        """Enhanced chat command processing with AI capabilities"""
        message = user_message.lower().strip()
        
        # AI prediction commands
        if any(keyword in message for keyword in ['predict', 'forecast', 'future cost']):
            return self._handle_prediction_command(user_message)
        
        # Anomaly detection commands
        if any(keyword in message for keyword in ['anomaly', 'anomalies', 'unusual', 'spike']):
            return self._handle_anomaly_command(user_message)
        
        # Optimization recommendation commands
        if any(keyword in message for keyword in ['optimize', 'recommendations', 'savings', 'reduce cost']):
            return self._handle_optimization_command(user_message)
        
        # Smart tagging commands
        if any(keyword in message for keyword in ['suggest tags', 'auto tag', 'smart tag']):
            return self._handle_smart_tagging_command(user_message)
        
        # Intelligent alerts
        if any(keyword in message for keyword in ['alerts', 'warnings', 'issues']):
            return self._handle_alerts_command(user_message)
        
        # Existing command processing (maintain all current functionality)
        if 'tag' in message and ('resource' in message or 'all' in message):
            return self._handle_tagging_command(user_message)
        elif any(keyword in message for keyword in ['terminate', 'start', 'stop']) and 'instance' in message:
            return self._handle_stopped_instance_command(user_message)
        elif any(keyword in message for keyword in ['delete', 'attach']) and 'volume' in message:
            return self._handle_unused_volume_command(user_message)
        elif any(keyword in message for keyword in ['show', 'list', 'get']):
            return self._handle_query_command(user_message)
        else:
            return ChatResponse(
                success=False,
                message="I didn't understand that command. Try asking about costs, tagging, instances, volumes, predictions, anomalies, or optimizations.",
                data={"available_commands": [
                    "Predict costs: 'predict costs for next 3 months'",
                    "Detect anomalies: 'show cost anomalies'",
                    "Get recommendations: 'show optimization recommendations'",
                    "Smart tagging: 'suggest tags for untagged resources'",
                    "View alerts: 'show priority alerts'",
                    "Tag resources: 'tag resource [name] with Project=WebApp'",
                    "Manage instances: 'terminate stopped instance [id]'",
                    "Manage volumes: 'delete unused volume [id]'",
                    "Query data: 'show project costs'"
                ]}
            )
    
    def _handle_prediction_command(self, user_message: str) -> ChatResponse:
        """Handle cost prediction commands"""
        message = user_message.lower()
        
        # Extract months ahead (default to 3)
        months_ahead = 3
        import re
        month_match = re.search(r'(\d+)\s*months?', message)
        if month_match:
            months_ahead = int(month_match.group(1))
        
        # Extract provider if specified
        provider = None
        if 'aws' in message:
            provider = 'AWS'
        elif 'azure' in message:
            provider = 'Azure'
        elif 'gcp' in message:
            provider = 'GCP'
        
        predictions = self.predict_monthly_costs(provider, months_ahead)
        
        if not predictions:
            return ChatResponse(
                success=False,
                message="Unable to generate predictions. Insufficient historical data.",
                data=None
            )
        
        # Format response
        response_message = f"💡 **Cost Predictions for Next {months_ahead} Months**\n\n"
        
        if provider:
            response_message += f"**Provider:** {provider}\n\n"
        
        for month, prediction in predictions.items():
            response_message += f"**Month {month.split('_')[1]}:**\n"
            response_message += f"• Predicted Cost: ${prediction.predicted_cost:,.2f}\n"
            response_message += f"• Confidence Range: ${prediction.confidence_interval[0]:,.2f} - ${prediction.confidence_interval[1]:,.2f}\n"
            response_message += f"• Trend: {prediction.trend.title()}\n"
            response_message += f"• Recommendation: {prediction.recommendation}\n\n"
        
        return ChatResponse(
            success=True,
            message=response_message,
            data={"predictions": predictions}
        )
    
    def _handle_anomaly_command(self, user_message: str) -> ChatResponse:
        """Handle anomaly detection commands"""
        message = user_message.lower()
        
        # Extract sensitivity level
        sensitivity = "medium"
        if 'high' in message or 'sensitive' in message:
            sensitivity = "high"
        elif 'low' in message:
            sensitivity = "low"
        
        anomalies = self.detect_cost_anomalies(sensitivity)
        
        if not anomalies:
            return ChatResponse(
                success=True,
                message="✅ No cost anomalies detected. Your spending patterns appear normal.",
                data={"anomalies": []}
            )
        
        # Format response
        response_message = f"🔍 **Cost Anomalies Detected** (Sensitivity: {sensitivity.title()})\n\n"
        
        for i, anomaly in enumerate(anomalies[:5], 1):  # Show top 5 anomalies
            response_message += f"**{i}. {anomaly.resource_id}** ({anomaly.severity.upper()})\n"
            response_message += f"• Issue: {anomaly.description}\n"
            response_message += f"• Cost Impact: ${anomaly.cost_impact:.2f}\n"
            response_message += f"• Recommended Action: {anomaly.recommended_action}\n\n"
        
        if len(anomalies) > 5:
            response_message += f"... and {len(anomalies) - 5} more anomalies detected.\n"
        
        total_impact = sum(a.cost_impact for a in anomalies)
        response_message += f"**Total Cost Impact:** ${total_impact:.2f}\n"
        
        return ChatResponse(
            success=True,
            message=response_message,
            data={"anomalies": [a.__dict__ for a in anomalies]}
        )
    
    def _handle_optimization_command(self, user_message: str) -> ChatResponse:
        """Handle optimization recommendation commands"""
        message = user_message.lower()
        
        # Extract focus area
        focus_area = "all"
        if 'rightsizing' in message or 'right size' in message:
            focus_area = "rightsizing"
        elif 'storage' in message:
            focus_area = "storage"
        elif 'scheduling' in message or 'schedule' in message:
            focus_area = "scheduling"
        elif 'reserved' in message or 'purchasing' in message:
            focus_area = "purchasing"
        
        recommendations = self.generate_optimization_recommendations(focus_area)
        
        if not recommendations:
            return ChatResponse(
                success=True,
                message="✅ No immediate optimization opportunities found. Your resources appear well-optimized.",
                data={"recommendations": []}
            )
        
        # Format response
        response_message = f"🎯 **Optimization Recommendations**"
        if focus_area != "all":
            response_message += f" (Focus: {focus_area.title()})"
        response_message += "\n\n"
        
        total_savings = 0
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5 recommendations
            response_message += f"**{i}. {rec.category.title()} - {rec.resource_type}**\n"
            response_message += f"• Current Cost: ${rec.current_cost:.2f}/month\n"
            response_message += f"• Potential Savings: ${rec.potential_savings:.2f}/month\n"
            response_message += f"• Confidence: {rec.confidence_score:.0%}\n"
            response_message += f"• Effort: {rec.implementation_effort.title()}\n"
            response_message += f"• Description: {rec.description}\n\n"
            total_savings += rec.potential_savings
        
        response_message += f"**Total Potential Monthly Savings:** ${total_savings:.2f}\n"
        response_message += f"**Annual Savings Potential:** ${total_savings * 12:.2f}\n"
        
        return ChatResponse(
            success=True,
            message=response_message,
            data={"recommendations": [r.__dict__ for r in recommendations]}
        )
    
    def _handle_smart_tagging_command(self, user_message: str) -> ChatResponse:
        """Handle smart tagging suggestion commands"""
        suggestions = self.suggest_intelligent_tags()
        
        if not suggestions:
            return ChatResponse(
                success=True,
                message="✅ All resources appear to be properly tagged.",
                data={"suggestions": []}
            )
        
        # Format response
        response_message = f"🏷️ **Smart Tagging Suggestions**\n\n"
        
        high_confidence = [s for s in suggestions if s.confidence_score >= 0.7]
        medium_confidence = [s for s in suggestions if 0.4 <= s.confidence_score < 0.7]
        
        if high_confidence:
            response_message += "**High Confidence Suggestions:**\n"
            for suggestion in high_confidence[:5]:
                response_message += f"• **{suggestion.resource_id}**\n"
                for tag_key, tag_value in suggestion.suggested_tags.items():
                    response_message += f"  - {tag_key}: {tag_value}\n"
                response_message += f"  - Confidence: {suggestion.confidence_score:.0%}\n"
                response_message += f"  - Reasoning: {suggestion.reasoning}\n\n"
        
        if medium_confidence:
            response_message += "**Medium Confidence Suggestions:**\n"
            for suggestion in medium_confidence[:3]:
                response_message += f"• **{suggestion.resource_id}** (Confidence: {suggestion.confidence_score:.0%})\n"
                tags_str = ", ".join([f"{k}={v}" for k, v in suggestion.suggested_tags.items()])
                response_message += f"  - Suggested: {tags_str}\n\n"
        
        response_message += "💡 **To apply tags, use commands like:**\n"
        response_message += "`Tag resource [resource_name] with Project=WebApp,Environment=Production`\n"
        
        return ChatResponse(
            success=True,
            message=response_message,
            data={"suggestions": [s.__dict__ for s in suggestions]}
        )
    
    def _handle_alerts_command(self, user_message: str) -> ChatResponse:
        """Handle intelligent alerts commands"""
        message = user_message.lower()
        
        # Extract priority filter
        priority_filter = "all"
        if 'critical' in message:
            priority_filter = "critical"
        elif 'high' in message:
            priority_filter = "high"
        elif 'medium' in message:
            priority_filter = "medium"
        
        alerts = self.get_intelligent_alerts(priority_filter)
        
        if not alerts:
            return ChatResponse(
                success=True,
                message="✅ No priority alerts at this time. Your cloud costs are looking good!",
                data={"alerts": []}
            )
        
        # Format response
        response_message = f"🚨 **Intelligent Alerts**"
        if priority_filter != "all":
            response_message += f" ({priority_filter.title()} Priority)"
        response_message += "\n\n"
        
        for i, alert in enumerate(alerts[:10], 1):  # Show top 10 alerts
            priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            emoji = priority_emoji.get(alert["priority"], "🔵")
            
            response_message += f"**{i}. {emoji} {alert['title']}**\n"
            response_message += f"• {alert['message']}\n"
            response_message += f"• Action: {alert['action']}\n"
            
            if alert['cost_impact'] > 0:
                response_message += f"• Cost Impact: +${alert['cost_impact']:.2f}\n"
            elif alert['cost_impact'] < 0:
                response_message += f"• Savings Potential: ${abs(alert['cost_impact']):.2f}\n"
            
            response_message += "\n"
        
        return ChatResponse(
            success=True,
            message=response_message,
            data={"alerts": alerts}
        )

    # ============= MAINTAIN ALL EXISTING FUNCTIONALITY =============
    
    def load_data(self):
        """Load both CSV and JSON data sources"""
        self.cost_data_df = self._load_csv_data()
        self.dummy_data = self._load_json_data()
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load or generate CSV cost data with enhanced fields"""
        if not os.path.exists(Config.COST_DATA_FILE):
            self._generate_enhanced_csv_data()
        
        try:
            df = pd.read_csv(
                Config.COST_DATA_FILE,
                skipinitialspace=True,
                on_bad_lines='warn',
                engine='python'
            )
            
            # Enhanced validation for new fields - ADD MISSING COLUMNS IF THEY DON'T EXIST
            required_columns = ['provider', 'service', 'resource', 'usage_hours', 'cost_usd', 'date']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Add missing columns if they don't exist
            if 'tags' not in df.columns:
                df['tags'] = df.apply(self._generate_random_tags, axis=1)
            if 'region' not in df.columns:
                df['region'] = df['provider'].apply(self._generate_random_region)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error reading CSV data: {e}")
            return pd.DataFrame(columns=['provider', 'service', 'resource', 'usage_hours', 'cost_usd', 'date', 'tags', 'region'])
    
    def _generate_random_tags(self, row) -> str:
        """Generate random tags for resources"""
        tag_options = [
            "Environment=Production,Team=DevOps",
            "Environment=Development,Team=Frontend",
            "Environment=Staging,Team=Backend",
            "Project=WebApp,CostCenter=Engineering",
            "Project=DataPipeline,CostCenter=Analytics",
            "Department=Marketing,Team=Digital",
            ""  # Some resources have no tags (for audit purposes)
        ]
        return random.choice(tag_options)
    
    def _generate_random_region(self, provider: str) -> str:
        """Generate random regions based on provider"""
        regions = {
            'AWS': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
            'Azure': ['East US', 'West Europe', 'Southeast Asia', 'Central US'],
            'GCP': ['us-central1', 'europe-west1', 'asia-southeast1', 'us-west1']
        }
        return random.choice(regions.get(provider, ['unknown']))
    
    def _load_json_data(self) -> Dict:
        """Load JSON dummy data"""
        try:
            if os.path.exists(Config.DUMMY_DATA_FILE):
                with open(Config.DUMMY_DATA_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading JSON data: {e}")
        return {}
    
    def _generate_enhanced_csv_data(self):
        """Generate enhanced sample CSV data with tags and regions"""
        data = []
        providers = ['AWS', 'Azure', 'GCP']
        services = {
            'AWS': ['EC2', 'S3', 'RDS', 'Lambda', 'CloudWatch', 'EBS', 'ELB'],
            'Azure': ['VM', 'BlobStorage', 'SQLDatabase', 'Functions', 'Monitor', 'Disk', 'LoadBalancer'],
            'GCP': ['ComputeEngine', 'CloudStorage', 'CloudSQL', 'CloudFunctions', 'Monitoring', 'PersistentDisk', 'LoadBalancer']
        }
        
        environments = ['Production', 'Development', 'Staging', 'Testing']
        teams = ['DevOps', 'Frontend', 'Backend', 'DataTeam', 'Security']
        projects = ['WebApp', 'MobileApp', 'DataPipeline', 'Analytics', 'Security']
        
        for i in range(100):  # More comprehensive sample data
            provider = random.choice(providers)
            service = random.choice(services[provider])
            resource = f"{service.lower()}-{i}"
            date = (datetime.date.today() - datetime.timedelta(days=random.randint(0, 180))).isoformat()
            usage = random.uniform(1, 300)
            cost = round(usage * random.uniform(0.01, 1.2), 2)
            
            # Generate realistic tags
            tags = ""
            if random.random() > 0.2:  # 80% of resources have tags
                env = random.choice(environments)
                team = random.choice(teams)
                project = random.choice(projects)
                tags = f"Environment={env},Team={team},Project={project}"
            
            region = self._generate_random_region(provider)
            
            data.append([provider, service, resource, usage, cost, date, tags, region])
        
        df = pd.DataFrame(data, columns=['provider', 'service', 'resource', 'usage_hours', 'cost_usd', 'date', 'tags', 'region'])
        df.to_csv(Config.COST_DATA_FILE, index=False)
    
    def generate_sample_budgets(self):
        """Generate sample budget data"""
        self.budgets = [
            BudgetInfo("Monthly AWS Budget", 10000.0, 8500.0, 85.0, "warning"),
            BudgetInfo("Azure Development", 5000.0, 3200.0, 64.0, "on_track"),
            BudgetInfo("GCP Production", 15000.0, 16200.0, 108.0, "exceeded"),
            BudgetInfo("Multi-Cloud Analytics", 8000.0, 7100.0, 88.75, "warning"),
        ]
    
    def generate_sample_ec2_data(self):
        """Generate sample EC2 instance data"""
        instance_states = ['running', 'stopped', 'terminated', 'pending']
        instance_types = ['t3.micro', 't3.small', 'm5.large', 'c5.xlarge', 'r5.2xlarge']
        projects = ['WebApp', 'MobileApp', 'DataPipeline', 'Analytics', 'Security']
        
        self.ec2_instances = []
        for i in range(20):
            self.ec2_instances.append({
                'instance_id': f'i-{random.randint(100000000000, 999999999999):012x}',
                'instance_type': random.choice(instance_types),
                'state': random.choice(instance_states),
                'region': random.choice(['us-east-1', 'us-west-2', 'eu-west-1']),
                'tags': f"Environment={random.choice(['Production', 'Development'])},Team={random.choice(['DevOps', 'Backend'])},Project={random.choice(projects)}" if random.random() > 0.3 else "",
                'monthly_cost': round(random.uniform(50, 500), 2)
            })

    def generate_sample_volumes(self):
        """Generate sample EBS volume data"""
        volume_states = ['in-use', 'available', 'creating', 'deleting']
        projects = ['WebApp', 'MobileApp', 'DataPipeline', 'Analytics', 'Security']
        
        self.volumes = []
        for i in range(15):
            self.volumes.append({
                'volume_id': f'vol-{random.randint(100000000000, 999999999999):012x}',
                'size': random.randint(10, 1000),
                'volume_type': random.choice(['gp3', 'gp2', 'io1', 'io2']),
                'state': random.choice(volume_states),
                'region': random.choice(['us-east-1', 'us-west-2', 'eu-west-1']),
                'tags': f"Environment={random.choice(['Production', 'Development'])},Team={random.choice(['DevOps', 'Backend'])},Project={random.choice(projects)}" if random.random() > 0.4 else "",
                'monthly_cost': round(random.uniform(10, 200), 2),
                'attached_instance': f'i-{random.randint(100000000000, 999999999999):012x}' if random.random() > 0.3 else None
            })
    
    def get_cost_by_time_range(self, time_range: TimeRange, provider: Optional[str] = None) -> Dict[str, float]:
        """Get costs filtered by time range"""
        if self.cost_data_df.empty:
            return {}
        
        df = self.cost_data_df.copy()
        if provider:
            df = df[df['provider'] == provider]
        
        now = datetime.datetime.now()
        
        if time_range == TimeRange.CURRENT_MONTH:
            start_date = now.replace(day=1)
            end_date = now
        elif time_range == TimeRange.PREVIOUS_MONTH:
            first_day_current = now.replace(day=1)
            last_day_previous = first_day_current - datetime.timedelta(days=1)
            start_date = last_day_previous.replace(day=1)
            end_date = last_day_previous
        else:
            start_date = now - datetime.timedelta(days=time_range.value)
            end_date = now
        
        filtered_df = df[
            (df['date'] >= start_date) & (df['date'] <= end_date)
        ]
        
        return filtered_df.groupby('provider')['cost_usd'].sum().to_dict()
    
    def get_cost_by_tags(self, tag_filters: List[str]) -> pd.DataFrame:
        """Get costs filtered by tags"""
        if self.cost_data_df.empty:
            return pd.DataFrame()
        
        # Check if tags column exists
        if 'tags' not in self.cost_data_df.columns:
            # Return empty DataFrame if no tags column
            return pd.DataFrame(columns=['provider', 'service', 'cost_usd'])
        
        df = self.cost_data_df.copy()
        
        for tag_filter in tag_filters:
            if '=' in tag_filter:
                key, value = tag_filter.split('=', 1)
                df = df[df['tags'].str.contains(f"{key}={value}", na=False)]
        
        if df.empty:
            return pd.DataFrame(columns=['provider', 'service', 'cost_usd'])
        
        return df.groupby(['provider', 'service'])['cost_usd'].sum().reset_index()

    # NEW FEATURE 1: Costs per project
    def get_cost_by_project(self, time_range: Optional[TimeRange] = None) -> Dict[str, Dict[str, float]]:
        """Get costs broken down by project"""
        if self.cost_data_df.empty:
            return {}
        
        df = self.cost_data_df.copy()
        
        # Apply time range filter if specified
        if time_range:
            now = datetime.datetime.now()
            if time_range == TimeRange.CURRENT_MONTH:
                start_date = now.replace(day=1)
                end_date = now
            elif time_range == TimeRange.PREVIOUS_MONTH:
                first_day_current = now.replace(day=1)
                last_day_previous = first_day_current - datetime.timedelta(days=1)
                start_date = last_day_previous.replace(day=1)
                end_date = last_day_previous
            else:
                start_date = now - datetime.timedelta(days=time_range.value)
                end_date = now
            
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Check if tags column exists
        if 'tags' not in df.columns:
            return {"No Projects Found": {"total_cost": df['cost_usd'].sum()}}
        
        project_costs = {}
        
        for _, row in df.iterrows():
            tags = row.get('tags', '')
            project = self._extract_project_from_tags(tags)
            
            if project not in project_costs:
                project_costs[project] = {
                    'total_cost': 0.0,
                    'providers': {},
                    'services': {},
                    'resource_count': 0
                }
            
            cost = row['cost_usd']
            provider = row['provider']
            service = row['service']
            
            project_costs[project]['total_cost'] += cost
            project_costs[project]['resource_count'] += 1
            
            # Track by provider
            if provider not in project_costs[project]['providers']:
                project_costs[project]['providers'][provider] = 0.0
            project_costs[project]['providers'][provider] += cost
            
            # Track by service
            if service not in project_costs[project]['services']:
                project_costs[project]['services'][service] = 0.0
            project_costs[project]['services'][service] += cost
        
        return project_costs

    def _extract_project_from_tags(self, tags: str) -> str:
        """Extract project name from tags string"""
        if not tags or pd.isna(tags):
            return "Untagged"
        
        # Look for Project=value pattern
        project_match = re.search(r'Project=([^,]+)', tags)
        if project_match:
            return project_match.group(1)
        
        return "Untagged"

    # NEW FEATURE 2: Interactive tagging via chat prompts
    def process_chat_command(self, user_message: str) -> ChatResponse:
        """Process user chat commands for tagging and resource management"""
        message = user_message.lower().strip()
        
        # Handle tagging commands
        if "tag" in message and ("resource" in message or "untagged" in message):
            return self._handle_tagging_command(user_message)
        
        # Handle stopped instance commands
        elif "stop" in message and "instance" in message:
            return self._handle_stopped_instance_command(user_message)
        
        # Handle unused volume commands  
        elif "volume" in message and ("unused" in message or "manage" in message):
            return self._handle_unused_volume_command(user_message)
        
        # Handle general resource queries
        elif "show" in message or "list" in message:
            return self._handle_query_command(user_message)
        
        else:
            return ChatResponse(
                success=False,
                message="I didn't understand that command. Try:\n" +
                       "- 'Tag resource [resource_name] with Project=ProjectName,Environment=Production'\n" +
                       "- 'Stop instance [instance_id]'\n" +
                       "- 'Delete unused volume [volume_id]'\n" +
                       "- 'Show untagged resources'"
            )

    def _handle_tagging_command(self, user_message: str) -> ChatResponse:
        """Handle resource tagging commands"""
        # Parse tagging command: "Tag resource [resource_name] with [tags]"
        tag_pattern = r'tag\s+resource\s+([^\s]+)\s+with\s+(.+)'
        match = re.search(tag_pattern, user_message.lower())
        
        if not match:
            # Try alternative pattern for bulk tagging
            bulk_pattern = r'tag\s+all\s+untagged\s+(.+?)\s+with\s+(.+)'
            bulk_match = re.search(bulk_pattern, user_message.lower())
            
            if bulk_match:
                resource_type = bulk_match.group(1)
                tags = bulk_match.group(2)
                return self._bulk_tag_resources(resource_type, tags)
            
            return ChatResponse(
                success=False,
                message="Please use format: 'Tag resource [resource_name] with [tags]' or 'Tag all untagged [service] with [tags]'"
            )
        
        resource_name = match.group(1)
        tags = match.group(2)
        
        return self._tag_resource(resource_name, tags)

    def _tag_resource(self, resource_name: str, tags: str) -> ChatResponse:
        """Tag a specific resource"""
        if self.cost_data_df.empty:
            return ChatResponse(success=False, message="No cost data available")
        
        # Find the resource
        resource_mask = self.cost_data_df['resource'].str.contains(resource_name, case=False, na=False)
        matching_resources = self.cost_data_df[resource_mask]
        
        if matching_resources.empty:
            return ChatResponse(
                success=False, 
                message=f"Resource '{resource_name}' not found"
            )
        
        # Update tags for all matching resources
        updated_count = 0
        for idx in matching_resources.index:
            current_tags = self.cost_data_df.at[idx, 'tags'] or ""
            
            # Merge new tags with existing ones
            new_tags = self._merge_tags(current_tags, tags)
            self.cost_data_df.at[idx, 'tags'] = new_tags
            updated_count += 1
        
        # Save updated data
        try:
            self.cost_data_df.to_csv(Config.COST_DATA_FILE, index=False)
            # Force reload data to ensure consistency
            self.load_data()
        except Exception as e:
            return ChatResponse(success=False, message=f"Failed to save changes: {e}")
        
        return ChatResponse(
            success=True,
            message=f"Successfully tagged {updated_count} resource(s) matching '{resource_name}' with: {tags}",
            data={"updated_count": updated_count, "tags": tags}
        )

    def _bulk_tag_resources(self, resource_type: str, tags: str) -> ChatResponse:
        """Tag all untagged resources of a specific type"""
        if self.cost_data_df.empty:
            return ChatResponse(success=False, message="No cost data available")
        
        # Find untagged resources of the specified type
        service_mask = self.cost_data_df['service'].str.contains(resource_type, case=False, na=False)
        untagged_mask = (self.cost_data_df['tags'].isna()) | (self.cost_data_df['tags'] == '')
        target_resources = self.cost_data_df[service_mask & untagged_mask]
        
        if target_resources.empty:
            return ChatResponse(
                success=False,
                message=f"No untagged {resource_type} resources found"
            )
        
        # Update tags and calculate cost impact
        updated_count = 0
        total_cost_allocated = 0
        for idx in target_resources.index:
            self.cost_data_df.at[idx, 'tags'] = tags
            updated_count += 1
            total_cost_allocated += self.cost_data_df.at[idx, 'cost_usd']
        
        # Save updated data
        try:
            self.cost_data_df.to_csv(Config.COST_DATA_FILE, index=False)
            # Force reload data to ensure consistency
            self.load_data()
        except Exception as e:
            return ChatResponse(success=False, message=f"Failed to save changes: {e}")
        
        return ChatResponse(
            success=True,
            message=f"Successfully tagged {updated_count} untagged {resource_type} resources with: {tags}. Total cost allocated: ${total_cost_allocated:.2f}",
            data={
                "updated_count": updated_count, 
                "resource_type": resource_type, 
                "tags": tags,
                "total_cost_allocated": total_cost_allocated
            }
        )

    def _merge_tags(self, existing_tags: str, new_tags: str) -> str:
        """Merge new tags with existing tags, avoiding duplicates"""
        if not existing_tags:
            return new_tags
        
        existing_dict = {}
        new_dict = {}
        
        # Parse existing tags
        for tag in existing_tags.split(','):
            if '=' in tag:
                key, value = tag.split('=', 1)
                existing_dict[key.strip()] = value.strip()
        
        # Parse new tags
        for tag in new_tags.split(','):
            if '=' in tag:
                key, value = tag.split('=', 1)
                new_dict[key.strip()] = value.strip()
        
        # Merge (new tags override existing ones)
        existing_dict.update(new_dict)
        
        # Convert back to string
        return ','.join([f"{k}={v}" for k, v in existing_dict.items()])

    # NEW FEATURE 3: Interactive stopped instance management
    def _handle_stopped_instance_command(self, user_message: str) -> ChatResponse:
        """Handle stopped instance management commands"""
        if "terminate" in user_message.lower() or "delete" in user_message.lower():
            # Extract instance ID
            instance_pattern = r'i-[a-f0-9]+'
            match = re.search(instance_pattern, user_message)
            
            if match:
                instance_id = match.group(0)
                return self._terminate_instance(instance_id)
            else:
                return ChatResponse(
                    success=False,
                    message="Please specify an instance ID (format: i-xxxxxxxxx)"
                )
        
        elif "start" in user_message.lower():
            instance_pattern = r'i-[a-f0-9]+'
            match = re.search(instance_pattern, user_message)
            
            if match:
                instance_id = match.group(0)
                return self._start_instance(instance_id)
            else:
                return ChatResponse(
                    success=False,
                    message="Please specify an instance ID to start"
                )
        
        else:
            return self._list_stopped_instances()

    def _terminate_instance(self, instance_id: str) -> ChatResponse:
        """Terminate a stopped instance"""
        instance = None
        for i, inst in enumerate(self.ec2_instances):
            if inst['instance_id'] == instance_id:
                instance = inst
                instance_index = i
                break
        
        if not instance:
            return ChatResponse(
                success=False,
                message=f"Instance {instance_id} not found"
            )
        
        if instance['state'] != 'stopped':
            return ChatResponse(
                success=False,
                message=f"Instance {instance_id} is {instance['state']}, not stopped. Only stopped instances can be terminated."
            )
        
        # Terminate the instance
        self.ec2_instances[instance_index]['state'] = 'terminated'
        monthly_savings = instance.get('monthly_cost', 0)
        
        return ChatResponse(
            success=True,
            message=f"Instance {instance_id} has been terminated. Monthly savings: ${monthly_savings:.2f}",
            data={
                "instance_id": instance_id,
                "monthly_savings": monthly_savings,
                "previous_state": "stopped"
            }
        )

    def _start_instance(self, instance_id: str) -> ChatResponse:
        """Start a stopped instance"""
        instance = None
        for i, inst in enumerate(self.ec2_instances):
            if inst['instance_id'] == instance_id:
                instance = inst
                instance_index = i
                break
        
        if not instance:
            return ChatResponse(
                success=False,
                message=f"Instance {instance_id} not found"
            )
        
        if instance['state'] != 'stopped':
            return ChatResponse(
                success=False,
                message=f"Instance {instance_id} is {instance['state']}, not stopped."
            )
        
        # Start the instance
        self.ec2_instances[instance_index]['state'] = 'running'
        monthly_cost = instance.get('monthly_cost', 0)
        
        return ChatResponse(
            success=True,
            message=f"Instance {instance_id} has been started. Monthly cost: ${monthly_cost:.2f}",
            data={
                "instance_id": instance_id,
                "monthly_cost": monthly_cost,
                "new_state": "running"
            }
        )

    def _list_stopped_instances(self) -> ChatResponse:
        """List all stopped instances"""
        stopped_instances = [inst for inst in self.ec2_instances if inst['state'] == 'stopped']
        
        if not stopped_instances:
            return ChatResponse(
                success=True,
                message="No stopped instances found",
                data={"stopped_instances": []}
            )
        
        total_potential_savings = sum(inst.get('monthly_cost', 0) for inst in stopped_instances)
        
        instance_details = []
        for inst in stopped_instances:
            instance_details.append({
                "instance_id": inst['instance_id'],
                "instance_type": inst['instance_type'],
                "region": inst['region'],
                "monthly_cost": inst.get('monthly_cost', 0),
                "tags": inst.get('tags', 'No tags')
            })
        
        message = f"Found {len(stopped_instances)} stopped instances. "
        message += f"Total potential monthly savings if terminated: ${total_potential_savings:.2f}"
        
        return ChatResponse(
            success=True,
            message=message,
            data={
                "stopped_instances": instance_details,
                "total_potential_savings": total_potential_savings
            }
        )

    # NEW FEATURE 4: Interactive unused volume management
    def _handle_unused_volume_command(self, user_message: str) -> ChatResponse:
        """Handle unused volume management commands"""
        if "delete" in user_message.lower() or "remove" in user_message.lower():
            # Extract volume ID
            volume_pattern = r'vol-[a-f0-9]+'
            match = re.search(volume_pattern, user_message)
            
            if match:
                volume_id = match.group(0)
                return self._delete_volume(volume_id)
            else:
                return ChatResponse(
                    success=False,
                    message="Please specify a volume ID (format: vol-xxxxxxxxx)"
                )
        
        elif "attach" in user_message.lower():
            volume_pattern = r'vol-[a-f0-9]+'
            instance_pattern = r'i-[a-f0-9]+'
            vol_match = re.search(volume_pattern, user_message)
            inst_match = re.search(instance_pattern, user_message)
            
            if vol_match and inst_match:
                volume_id = vol_match.group(0)
                instance_id = inst_match.group(0)
                return self._attach_volume(volume_id, instance_id)
            else:
                return ChatResponse(
                    success=False,
                    message="Please specify both volume ID and instance ID"
                )
        
        else:
            return self._list_unused_volumes()

    def _delete_volume(self, volume_id: str) -> ChatResponse:
        """Delete an unused volume"""
        volume = None
        for i, vol in enumerate(self.volumes):
            if vol['volume_id'] == volume_id:
                volume = vol
                volume_index = i
                break
        
        if not volume:
            return ChatResponse(
                success=False,
                message=f"Volume {volume_id} not found"
            )
        
        if volume['state'] != 'available':
            return ChatResponse(
                success=False,
                message=f"Volume {volume_id} is {volume['state']}, not available. Only available volumes can be deleted."
            )
        
        # Delete the volume
        monthly_savings = volume.get('monthly_cost', 0)
        del self.volumes[volume_index]
        
        return ChatResponse(
            success=True,
            message=f"Volume {volume_id} ({volume['size']}GB) has been deleted. Monthly savings: ${monthly_savings:.2f}",
            data={
                "volume_id": volume_id,
                "size": volume['size'],
                "monthly_savings": monthly_savings
            }
        )

    def _attach_volume(self, volume_id: str, instance_id: str) -> ChatResponse:
        """Attach a volume to an instance"""
        volume = None
        for i, vol in enumerate(self.volumes):
            if vol['volume_id'] == volume_id:
                volume = vol
                volume_index = i
                break
        
        if not volume:
            return ChatResponse(
                success=False,
                message=f"Volume {volume_id} not found"
            )
        
        if volume['state'] != 'available':
            return ChatResponse(
                success=False,
                message=f"Volume {volume_id} is {volume['state']}, not available for attachment."
            )
        
        # Check if instance exists
        instance_exists = any(inst['instance_id'] == instance_id for inst in self.ec2_instances)
        if not instance_exists:
            return ChatResponse(
                success=False,
                message=f"Instance {instance_id} not found"
            )
        
        # Attach the volume
        self.volumes[volume_index]['state'] = 'in-use'
        self.volumes[volume_index]['attached_instance'] = instance_id
        
        return ChatResponse(
            success=True,
            message=f"Volume {volume_id} has been attached to instance {instance_id}",
            data={
                "volume_id": volume_id,
                "instance_id": instance_id,
                "size": volume['size']
            }
        )

    def _list_unused_volumes(self) -> ChatResponse:
        """List all unused volumes"""
        unused_volumes = [vol for vol in self.volumes if vol['state'] == 'available']
        
        if not unused_volumes:
            return ChatResponse(
                success=True,
                message="No unused volumes found",
                data={"unused_volumes": []}
            )
        
        total_potential_savings = sum(vol.get('monthly_cost', 0) for vol in unused_volumes)
        total_size = sum(vol['size'] for vol in unused_volumes)
        
        volume_details = []
        for vol in unused_volumes:
            volume_details.append({
                "volume_id": vol['volume_id'],
                "size": vol['size'],
                "volume_type": vol['volume_type'],
                "region": vol['region'],
                "monthly_cost": vol.get('monthly_cost', 0),
                "tags": vol.get('tags', 'No tags')
            })
        
        message = f"Found {len(unused_volumes)} unused volumes ({total_size}GB total). "
        message += f"Total potential monthly savings if deleted: ${total_potential_savings:.2f}"
        
        return ChatResponse(
            success=True,
            message=message,
            data={
                "unused_volumes": volume_details,
                "total_potential_savings": total_potential_savings,
                "total_size": total_size
            }
        )

    def _handle_query_command(self, user_message: str) -> ChatResponse:
        """Handle query commands"""
        message = user_message.lower()
        
        if "untagged" in message:
            return self._show_untagged_resources()
        elif "stopped" in message and "instance" in message:
            return self._list_stopped_instances()
        elif "unused" in message and "volume" in message:
            return self._list_unused_volumes()
        elif "project" in message and "cost" in message:
            # Check if asking for specific project
            project_match = re.search(r'project\s+([a-zA-Z0-9_-]+)', user_message)
            if project_match:
                project_name = project_match.group(1)
                return self._show_specific_project_costs(project_name)
            else:
                return self._show_project_costs()
        else:
            return ChatResponse(
                success=False,
                message="Available queries: 'show untagged resources', 'show stopped instances', 'show unused volumes', 'show project costs', 'show costs for project [ProjectName]'"
            )

    def _show_untagged_resources(self) -> ChatResponse:
        """Show all untagged resources"""
        if self.cost_data_df.empty:
            return ChatResponse(success=False, message="No cost data available")
        
        if 'tags' not in self.cost_data_df.columns:
            untagged_count = len(self.cost_data_df)
            total_cost = self.cost_data_df['cost_usd'].sum()
        else:
            untagged_mask = (self.cost_data_df['tags'].isna()) | (self.cost_data_df['tags'] == '')
            untagged_resources = self.cost_data_df[untagged_mask]
            untagged_count = len(untagged_resources)
            total_cost = untagged_resources['cost_usd'].sum() if not untagged_resources.empty else 0
        
        return ChatResponse(
            success=True,
            message=f"Found {untagged_count} untagged resources with total cost of ${total_cost:.2f}",
            data={
                "untagged_count": untagged_count,
                "total_cost": total_cost
            }
        )

    def _show_project_costs(self) -> ChatResponse:
        """Show costs breakdown by project"""
        project_costs = self.get_cost_by_project()
        
        if not project_costs:
            return ChatResponse(
                success=False,
                message="No project cost data available"
            )
        
        # Sort projects by total cost
        sorted_projects = sorted(project_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)
        
        message = "Project Cost Breakdown:\n"
        total_all_projects = sum(data['total_cost'] for data in project_costs.values())
        
        for project, data in sorted_projects:
            percentage = (data['total_cost'] / total_all_projects) * 100 if total_all_projects > 0 else 0
            message += f"• {project}: ${data['total_cost']:.2f} ({percentage:.1f}%) - {data['resource_count']} resources\n"
        
        return ChatResponse(
            success=True,
            message=message,
            data={
                "project_costs": dict(sorted_projects),
                "total_cost": total_all_projects
            }
        )
    
    def _show_specific_project_costs(self, project_name: str) -> ChatResponse:
        """Show costs for a specific project"""
        project_costs = self.get_cost_by_project()
        
        if not project_costs:
            return ChatResponse(
                success=False,
                message="No project cost data available"
            )
        
        # Find project (case-insensitive)
        matching_project = None
        for proj_name, proj_data in project_costs.items():
            if proj_name.lower() == project_name.lower():
                matching_project = (proj_name, proj_data)
                break
        
        if not matching_project:
            available_projects = list(project_costs.keys())
            return ChatResponse(
                success=False,
                message=f"Project '{project_name}' not found. Available projects: {', '.join(available_projects)}"
            )
        
        proj_name, proj_data = matching_project
        
        message = f"Cost Analysis for Project '{proj_name}':\n"
        message += f"• Total Cost: ${proj_data['total_cost']:.2f}\n"
        message += f"• Resource Count: {proj_data['resource_count']}\n"
        
        if proj_data['providers']:
            message += f"• Top Provider: {max(proj_data['providers'].items(), key=lambda x: x[1])[0]}\n"
        
        if proj_data['services']:
            top_services = sorted(proj_data['services'].items(), key=lambda x: x[1], reverse=True)[:3]
            message += f"• Top Services: {', '.join([f'{svc} (${cost:.2f})' for svc, cost in top_services])}\n"
        
        return ChatResponse(
            success=True,
            message=message,
            data={
                "project_name": proj_name,
                "project_data": proj_data
            }
        )
    
    def get_cost_trends_six_months(self) -> pd.DataFrame:
        """Get cost trends for the last 6 months"""
        if self.cost_data_df.empty:
            return pd.DataFrame()
        
        six_months_ago = datetime.datetime.now() - datetime.timedelta(days=180)
        recent_data = self.cost_data_df[self.cost_data_df['date'] >= six_months_ago]
        
        # Group by month and provider
        recent_data['month'] = recent_data['date'].dt.to_period('M')
        monthly_costs = recent_data.groupby(['month', 'provider'])['cost_usd'].sum().reset_index()
        monthly_costs['month'] = monthly_costs['month'].astype(str)
        
        return monthly_costs
    
    def get_budget_status(self) -> List[BudgetInfo]:
        """Get current budget status"""
        return self.budgets
    
    def get_ec2_instance_summary(self) -> Dict[str, int]:
        """Get EC2 instance summary by state"""
        summary = {}
        for instance in self.ec2_instances:
            state = instance['state']
            summary[state] = summary.get(state, 0) + 1
        return summary
    
    def perform_finops_audit(self) -> ResourceAudit:
        """Perform comprehensive FinOps audit"""
        if self.cost_data_df.empty:
            return ResourceAudit([], [], [], [], self.budgets)
        
        # Find untagged resources - handle missing tags column safely
        try:
            if 'tags' in self.cost_data_df.columns:
                untagged = self.cost_data_df[
                    (self.cost_data_df['tags'].isna()) | (self.cost_data_df['tags'] == '')
                ].to_dict('records')
            else:
                # If no tags column, consider all resources as untagged
                untagged = self.cost_data_df.to_dict('records')
        except Exception as e:
            print(f"Error in audit: {e}")
            untagged = []
        
        # Find stopped instances (simulated)
        stopped_instances = [inst for inst in self.ec2_instances if inst['state'] == 'stopped']
        
        # Find unused volumes 
        unused_volumes = [vol for vol in self.volumes if vol['state'] == 'available']
        
        # Simulate unused EIPs
        unused_eips = [
            {'allocation_id': f'eip-{i:08x}', 'region': 'us-west-2'}
            for i in range(random.randint(1, 5))
        ]
        
        return ResourceAudit(
            untagged_resources=untagged,
            stopped_instances=stopped_instances,
            unused_volumes=unused_volumes,
            unused_eips=unused_eips,
            budget_alerts=[b for b in self.budgets if b.status in ['warning', 'exceeded']]
        )
    
    def export_data(self, report_name: str, report_type: List[str], directory: str = ".") -> List[str]:
        """Export data in specified formats"""
        exported_files = []
        
        # Prepare comprehensive data for export
        export_data = {
            'cost_summary': self.get_cost_summary_by_provider(),
            'service_costs': self.get_cost_summary_by_service(),
            'monthly_trends': self.get_cost_trends_six_months().to_dict('records'),
            'budget_status': [b.__dict__ for b in self.budgets],
            'ec2_summary': self.get_ec2_instance_summary(),
            'audit_results': self.perform_finops_audit().__dict__
        }
        
        for format_type in report_type:
            filename = f"{report_name}.{format_type}"
            filepath = os.path.join(directory, filename)
            
            if format_type.lower() == 'json':
                # Convert any non-serializable objects to dict
                serializable_data = self._make_serializable(export_data)
                with open(filepath, 'w') as f:
                    json.dump(serializable_data, f, indent=2, default=str)
                exported_files.append(filepath)
                
            elif format_type.lower() == 'csv':
                # Export main cost data as CSV
                if not self.cost_data_df.empty:
                    self.cost_data_df.to_csv(filepath, index=False)
                    exported_files.append(filepath)
        
        return exported_files
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    # ============= LEGACY COMPATIBILITY METHODS =============
    
    def get_cost_summary_by_provider(self) -> Dict[str, float]:
        """Get total cost by cloud provider"""
        if self.cost_data_df.empty:
            return {}
        return self.cost_data_df.groupby('provider')['cost_usd'].sum().to_dict()
    
    def get_cost_summary_by_service(self) -> Dict[str, float]:
        """Get total cost by service"""
        if self.cost_data_df.empty:
            return {}
        return self.cost_data_df.groupby('service')['cost_usd'].sum().to_dict()
    
    def get_cost_trends(self, days: int = 30) -> pd.DataFrame:
        """Get cost trends over the last N days"""
        if self.cost_data_df.empty:
            return pd.DataFrame()
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_data = self.cost_data_df[self.cost_data_df['date'] >= cutoff_date]
        
        return recent_data.groupby(['date', 'provider'])['cost_usd'].sum().reset_index()
    
    def get_top_services(self, provider: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Get top services by cost, optionally filtered by provider"""
        if self.cost_data_df.empty:
            return []
        
        df = self.cost_data_df
        if provider:
            df = df[df['provider'] == provider]
        
        top_services = df.groupby('service')['cost_usd'].sum().sort_values(ascending=False).head(limit)
        return [{'service': service, 'total_cost': cost} for service, cost in top_services.items()]
    
    def get_optimization_insights(self) -> List[str]:
        """Generate enhanced cost optimization insights"""
        insights = []
        
        if self.cost_data_df.empty:
            return ["No cost data available for analysis."]
        
        # High-cost services analysis
        top_services = self.get_top_services(limit=3)
        for service_data in top_services:
            service = service_data['service']
            cost = service_data['total_cost']
            
            if cost > 100:
                if service in ['EC2', 'VM', 'ComputeEngine']:
                    insights.append(f"🔍 {service} costs are high (${cost:.2f}). Consider rightsizing instances or using spot/preemptible instances.")
                elif service in ['S3', 'BlobStorage', 'CloudStorage']:
                    insights.append(f"📦 {service} costs are significant (${cost:.2f}). Review storage classes and lifecycle policies.")
                elif service in ['RDS', 'SQLDatabase', 'CloudSQL']:
                    insights.append(f"🗄️ {service} costs are notable (${cost:.2f}). Consider reserved instances or database optimization.")
        
        # Budget analysis
        for budget in self.budgets:
            if budget.status == 'exceeded':
                insights.append(f"🚨 Budget Alert: {budget.name} exceeded by {budget.percentage_used - 100:.1f}%")
            elif budget.status == 'warning':
                insights.append(f"⚠️ Budget Warning: {budget.name} at {budget.percentage_used:.1f}% of limit")
        
        # Audit insights
        audit = self.perform_finops_audit()
        if audit.untagged_resources:
            insights.append(f"🏷️ Found {len(audit.untagged_resources)} untagged resources. Implement consistent tagging for better cost allocation.")
        
        if audit.stopped_instances:
            insights.append(f"⏹️ Found {len(audit.stopped_instances)} stopped EC2 instances. Consider terminating if no longer needed.")
        
        return insights if insights else ["Your costs appear to be well-distributed. Continue monitoring for optimization opportunities."] 

    def get_cost_summary_by_provider(self):
        """Get cost summary grouped by cloud provider."""
        try:
            # Get all resources
            resources = self.get_all_resources()
            
            # Initialize provider costs
            provider_costs = {
                'AWS': 0.0,
                'Azure': 0.0,
                'GCP': 0.0
            }
            
            # Calculate costs by provider
            for resource in resources:
                provider = resource.get('provider', '')
                # Handle both string and numeric cost values
                cost = resource.get('cost', 0)
                if isinstance(cost, str):
                    try:
                        cost = float(cost.replace('$', '').replace(',', ''))
                    except ValueError:
                        cost = 0.0
                elif isinstance(cost, (int, float)):
                    cost = float(cost)
                else:
                    cost = 0.0
                
                if provider in provider_costs:
                    provider_costs[provider] += cost
            
            # Round to 2 decimal places
            provider_costs = {k: round(v, 2) for k, v in provider_costs.items()}
            
            return provider_costs
            
        except Exception as e:
            print(f"Error getting cost summary by provider: {str(e)}")
            return {
                'AWS': 0.0,
                'Azure': 0.0,
                'GCP': 0.0
            }

    def get_provider_cost_breakdown(self, provider):
        """Get detailed cost breakdown for a specific provider."""
        try:
            resources = self.get_all_resources()
            service_costs = {}
            total_cost = 0.0
            
            for resource in resources:
                if resource.get('provider') == provider:
                    service = resource.get('service', 'Unknown')
                    cost = resource.get('cost', 0)
                    
                    # Handle both string and numeric cost values
                    if isinstance(cost, str):
                        try:
                            cost = float(cost.replace('$', '').replace(',', ''))
                        except ValueError:
                            cost = 0.0
                    elif isinstance(cost, (int, float)):
                        cost = float(cost)
                    else:
                        cost = 0.0
                    
                    service_costs[service] = service_costs.get(service, 0) + cost
                    total_cost += cost
            
            # Sort services by cost
            sorted_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'total_cost': round(total_cost, 2),
                'service_breakdown': dict(sorted_services)
            }
            
        except Exception as e:
            print(f"Error getting provider cost breakdown: {str(e)}")
            return {
                'total_cost': 0.0,
                'service_breakdown': {}
            }

    def get_all_resources(self):
        """Get all resources from all providers."""
        try:
            all_resources = []
            
            if self.cost_data_df is not None and not self.cost_data_df.empty:
                # Convert DataFrame to list of dictionaries
                for _, row in self.cost_data_df.iterrows():
                    resource = {
                        'resource_id': row.get('resource', ''),
                        'provider': row.get('provider', ''),
                        'service': row.get('service', ''),
                        'cost': row.get('cost_usd', 0),
                        'region': row.get('region', ''),
                        'tags': row.get('tags', ''),
                        'date': row.get('date', ''),
                        'instance_type': row.get('instance_type', ''),
                        'usage_hours': row.get('usage_hours', 0)
                    }
                    all_resources.append(resource)
            
            return all_resources
            
        except Exception as e:
            print(f"Error getting all resources: {str(e)}")
            return []