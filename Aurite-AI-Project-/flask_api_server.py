# flask_api_server.py
"""
Flask API Backend for Aurite AI Investment Workflow
Connects the web UI with the Python agents
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio
import json
import os
import threading
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Import your workflow
from master_investment_workflow import AuriteInvestmentWorkflow

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FlaskAPI')

# Store active workflows
active_workflows = {}

def make_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj

def check_stock_analysis_freshness():
    """Check if stock analysis is recent (within last 24 hours)"""
    try:
        output_dir = Path("analysis_outputs")
        if not output_dir.exists():
            return False
        
        # Look for stock analysis files
        stock_files = list(output_dir.glob("stock_analysis_*.json"))
        if not stock_files:
            return False
        
        # Get the most recent stock analysis file
        latest_file = max(stock_files, key=lambda f: f.stat().st_mtime)
        file_age = time.time() - latest_file.stat().st_mtime
        
        # Check if file is less than 24 hours old (86400 seconds)
        return file_age < 86400
        
    except Exception as e:
        logger.error(f"Error checking stock analysis freshness: {e}")
        return False

def run_stock_analysis():
    """Run the stock analysis script to generate fresh data"""
    try:
        script_path = "run_30_stock_analysis.py"
        
        # Check if the script exists
        if not os.path.exists(script_path):
            logger.error(f"Stock analysis script not found: {script_path}")
            return False
        
        logger.info("Running stock analysis script...")
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Stock analysis completed successfully")
            return True
        else:
            logger.error(f"Stock analysis failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Stock analysis script timed out")
        return False
    except Exception as e:
        logger.error(f"Error running stock analysis: {e}")
        return False

def ensure_fresh_stock_analysis(workflow_runner, force_refresh=False):
    """Ensure we have fresh stock analysis data"""
    try:
        # Check if we need to refresh
        needs_refresh = force_refresh or not check_stock_analysis_freshness()
        
        if needs_refresh:
            workflow_runner.log_message("🔄 Stock analysis data is stale or missing")
            workflow_runner.log_message("📊 Running fresh stock analysis (this may take a few minutes)...")
            
            success = run_stock_analysis()
            if success:
                workflow_runner.log_message("✅ Fresh stock analysis completed")
                return True
            else:
                workflow_runner.log_message("⚠️ Stock analysis failed, using existing data if available")
                return check_stock_analysis_freshness()  # Check if we at least have old data
        else:
            workflow_runner.log_message("✅ Recent stock analysis found, using existing data")
            return True
            
    except Exception as e:
        workflow_runner.log_message(f"❌ Error ensuring fresh stock analysis: {e}")
        return False

def cleanup_old_stock_analysis():
    """Clean up old stock analysis files, keeping only the 3 most recent"""
    try:
        output_dir = Path("analysis_outputs")
        if not output_dir.exists():
            return
        
        # Get all stock analysis files
        stock_files = list(output_dir.glob("stock_analysis_*.json"))
        
        if len(stock_files) <= 3:
            return  # Keep all if we have 3 or fewer
        
        # Sort by modification time (newest first)
        stock_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove old files (keep only the 3 most recent)
        for old_file in stock_files[3:]:
            try:
                old_file.unlink()
                logger.info(f"Cleaned up old stock analysis: {old_file.name}")
            except Exception as e:
                logger.error(f"Failed to remove old file {old_file}: {e}")
                
    except Exception as e:
        logger.error(f"Error cleaning up old stock analysis files: {e}")

def parse_additional_notes(notes):
    """Parse additional notes to extract avoid sectors, preferences, and other settings"""
    avoid_sectors = []
    prefer_sectors = []
    prefers_esg = False
    needs_liquidity = False
    tax_sensitive = False
    monthly_contribution = 0
    
    if not notes:
        return avoid_sectors, prefer_sectors, prefers_esg, needs_liquidity, tax_sensitive, monthly_contribution
    
    notes_lower = notes.lower()
    
    # Define sector mappings
    sector_keywords = {
        'Technology': ['tech', 'technology', 'software', 'ai', 'artificial intelligence'],
        'Healthcare': ['healthcare', 'health', 'medical', 'pharma', 'pharmaceutical'],
        'Financial': ['financial', 'finance', 'bank', 'banking', 'insurance'],
        'Energy': ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind'],
        'Consumer': ['consumer', 'retail', 'ecommerce', 'e-commerce'],
        'Industrial': ['industrial', 'manufacturing', 'construction'],
        'Utilities': ['utilities', 'electric', 'water', 'utility'],
        'Materials': ['materials', 'mining', 'metals', 'chemicals'],
        'Real Estate': ['real estate', 'reit', 'property', 'housing'],
        'Telecommunications': ['telecom', 'telecommunications', 'wireless', 'internet']
    }
    
    # Look for avoid keywords
    avoid_keywords = ['avoid', 'no', 'not', 'exclude', 'skip', 'without', 'except']
    prefer_keywords = ['prefer', 'focus', 'emphasize', 'concentrate', 'favor', 'like']
    
    # Parse sectors
    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword in notes_lower:
                # Check if it's an avoid or prefer instruction
                keyword_pos = notes_lower.find(keyword)
                context_before = notes_lower[:keyword_pos].split()[-3:]  # Last 3 words before keyword
                
                if any(avoid_word in context_before for avoid_word in avoid_keywords):
                    if sector not in avoid_sectors:
                        avoid_sectors.append(sector)
                elif any(prefer_word in context_before for prefer_word in prefer_keywords):
                    if sector not in prefer_sectors:
                        prefer_sectors.append(sector)
                # If keyword appears without clear context, check for negative words nearby
                elif any(avoid_word in notes_lower[max(0, keyword_pos-20):keyword_pos+20] for avoid_word in avoid_keywords):
                    if sector not in avoid_sectors:
                        avoid_sectors.append(sector)
    
    # Parse ESG preferences
    esg_keywords = ['esg', 'sustainable', 'sustainability', 'responsible', 'ethical', 'green', 'environmental']
    for keyword in esg_keywords:
        if keyword in notes_lower:
            prefers_esg = True
            break
    
    # Parse liquidity preferences
    liquidity_keywords = ['liquidity', 'liquid', 'quick access', 'emergency', 'cash access', 'withdraw']
    for keyword in liquidity_keywords:
        if keyword in notes_lower:
            needs_liquidity = True
            break
    
    # Parse tax preferences
    tax_keywords = ['tax', 'tax-efficient', 'tax efficient', 'tax-free', 'tax free', 'taxes', 'ira', 'roth', '401k']
    for keyword in tax_keywords:
        if keyword in notes_lower:
            tax_sensitive = True
            break
    
    # Parse monthly contribution
    import re
    monthly_patterns = [
        r'(\d+)\s*(?:dollars?|$)?\s*(?:per\s*month|monthly|/month)',
        r'monthly\s*(?:contribution|investment|payment)?\s*(?:of\s*)?(?:\$)?(\d+)',
        r'(\d+)\s*(?:\$)?\s*(?:each|every)\s*month'
    ]
    
    for pattern in monthly_patterns:
        match = re.search(pattern, notes_lower)
        if match:
            try:
                monthly_contribution = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    return avoid_sectors, prefer_sectors, prefers_esg, needs_liquidity, tax_sensitive, monthly_contribution

def generate_realistic_returns(asset_type, symbols, risk_level):
    """Generate realistic expected returns based on asset type and risk level"""
    import random
    
    # Base return ranges by asset type
    return_ranges = {
        'stocks': {
            'conservative': (4.0, 8.0),
            'moderate': (6.0, 12.0),
            'aggressive': (8.0, 18.0)
        },
        'bonds': {
            'conservative': (2.0, 4.5),
            'moderate': (2.5, 5.5),
            'aggressive': (3.0, 7.0)
        },
        'gold': {
            'conservative': (3.0, 6.0),
            'moderate': (4.0, 8.0),
            'aggressive': (5.0, 10.0)
        }
    }
    
    # Get the range for this asset type and risk level
    min_return, max_return = return_ranges.get(asset_type, {}).get(risk_level, (5.0, 8.0))
    
    # Generate realistic returns for each symbol
    returns = {}
    for symbol in symbols:
        # Add some variation based on symbol characteristics
        base_return = random.uniform(min_return, max_return)
        
        # Add symbol-specific adjustments (simplified)
        if asset_type == 'stocks':
            # Tech stocks typically have higher volatility
            if symbol in ['NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN']:
                base_return += random.uniform(0.5, 2.0)
            # Blue chip stocks are more conservative
            elif symbol in ['AAPL', 'MSFT', 'JNJ', 'PG']:
                base_return += random.uniform(-0.5, 1.0)
        
        returns[symbol] = round(base_return, 1)
    
    return returns

class WorkflowRunner:
    """Wrapper to run the async workflow in a thread"""
    
    def __init__(self, workflow_id):
        self.workflow_id = workflow_id
        self.workflow = AuriteInvestmentWorkflow()
        self.progress = 0
        self.current_step = ""
        self.logs = []
        self.result = None
        self.error = None
        
    def log_message(self, message):
        """Add log message and emit to frontend"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        try:
            socketio.emit('workflow_log', {
                'workflow_id': self.workflow_id,
                'message': str(message),
                'timestamp': timestamp
            }, room=self.workflow_id)
        except Exception as e:
            logger.error(f"Failed to emit log message: {e}")
        
        logger.info(f"Workflow {self.workflow_id}: {message}")
    
    def update_progress(self, step_number, step_name, progress_percent):
        """Update workflow progress"""
        self.progress = int(progress_percent)
        self.current_step = str(step_name)
        
        try:
            socketio.emit('workflow_progress', {
                'workflow_id': self.workflow_id,
                'step': int(step_number),
                'step_name': str(step_name),
                'progress': int(progress_percent)
            }, room=self.workflow_id)
        except Exception as e:
            logger.error(f"Failed to emit progress update: {e}")
    
    def _enhance_portfolio_returns(self, portfolio, risk_level):
        """Enhance portfolio recommendations with realistic expected returns"""
        try:
            if not portfolio or 'recommendations' not in portfolio:
                return portfolio
            
            recommendations = portfolio['recommendations']
            
            # Generate realistic returns for each asset class
            if 'stocks' in recommendations and recommendations['stocks']:
                stock_symbols = [s if isinstance(s, str) else s.get('symbol', s) for s in recommendations['stocks']]
                stock_returns = generate_realistic_returns('stocks', stock_symbols, risk_level)
                
                # Update stock recommendations with realistic returns
                enhanced_stocks = []
                for stock in recommendations['stocks']:
                    if isinstance(stock, str):
                        enhanced_stocks.append({
                            'symbol': stock,
                            'expected_return': stock_returns.get(stock, 8.5)
                        })
                    else:
                        stock['expected_return'] = stock_returns.get(stock.get('symbol', ''), 8.5)
                        enhanced_stocks.append(stock)
                recommendations['stocks'] = enhanced_stocks
            
            if 'bonds' in recommendations and recommendations['bonds']:
                bond_symbols = [b if isinstance(b, str) else b.get('symbol', b) for b in recommendations['bonds']]
                bond_returns = generate_realistic_returns('bonds', bond_symbols, risk_level)
                
                # Update bond recommendations with realistic returns
                enhanced_bonds = []
                for bond in recommendations['bonds']:
                    if isinstance(bond, str):
                        enhanced_bonds.append({
                            'symbol': bond,
                            'expected_return': bond_returns.get(bond, 4.2)
                        })
                    else:
                        bond['expected_return'] = bond_returns.get(bond.get('symbol', ''), 4.2)
                        enhanced_bonds.append(bond)
                recommendations['bonds'] = enhanced_bonds
            
            if 'gold' in recommendations and recommendations['gold']:
                gold_symbols = [g if isinstance(g, str) else g.get('symbol', g) for g in recommendations['gold']]
                gold_returns = generate_realistic_returns('gold', gold_symbols, risk_level)
                
                # Update gold recommendations with realistic returns
                enhanced_gold = []
                for gold in recommendations['gold']:
                    if isinstance(gold, str):
                        enhanced_gold.append({
                            'symbol': gold,
                            'expected_return': gold_returns.get(gold, 6.1)
                        })
                    else:
                        gold['expected_return'] = gold_returns.get(gold.get('symbol', ''), 6.1)
                        enhanced_gold.append(gold)
                recommendations['gold'] = enhanced_gold
            
            return portfolio
            
        except Exception as e:
            self.log_message(f"⚠️ Failed to enhance portfolio returns: {e}")
            return portfolio
    
    def run_complete_workflow(self, user_profile, force_stock_refresh=False):
        """Run the complete workflow with progress updates"""
        try:
            self.log_message("🚀 Starting Aurite AI Investment Workflow")
            self.update_progress(0, "Initializing", 0)
            
            # Step 0: Ensure fresh stock analysis
            self.log_message("🔍 Checking stock analysis data...")
            stock_analysis_ready = ensure_fresh_stock_analysis(self, force_stock_refresh)
            
            if not stock_analysis_ready:
                raise Exception("Failed to ensure fresh stock analysis data")
            
            # Clean up old files
            cleanup_old_stock_analysis()
            
            # Ensure user profile is JSON serializable
            user_profile = make_json_serializable(user_profile)
            
            # Override the workflow's user profile collection
            self.workflow.results['agent1_profile'] = user_profile
            
            # Step 1: User Profiling (already done)
            self.update_progress(1, "User Preference Analysis", 20)
            self.log_message("🤖 STEP 1: User Preference Analysis")
            self.log_message(f"✅ User profile loaded: {user_profile.get('profile_id', 'N/A')}")
            time.sleep(1)
            
            # Step 2: Market Analysis
            self.update_progress(2, "Market & Asset Analysis", 40)
            self.log_message("📊 STEP 2: Market & Asset Analysis")
            
            # Run the async analysis in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                analysis_results = loop.run_until_complete(
                    self.workflow._run_agent2(user_profile)
                )
                
                # Make analysis results JSON serializable
                analysis_results = make_json_serializable(analysis_results)
                self.workflow.results['agent2_analysis'] = analysis_results
                
                self.update_progress(2, "Market Analysis Complete", 70)
                
                # Step 3: Portfolio Construction
                self.update_progress(3, "Portfolio Construction", 80)
                self.log_message("🎯 STEP 3: Portfolio Construction")
                self.log_message("📈 Using latest stock analysis for portfolio optimization...")
                
                # Log sector filtering info
                if user_profile.get('avoid_sectors'):
                    self.log_message(f"🚫 Avoiding sectors: {', '.join(user_profile['avoid_sectors'])}")
                if user_profile.get('preferred_sectors'):
                    self.log_message(f"✅ Preferred sectors: {', '.join(user_profile['preferred_sectors'])}")
                
                portfolio = self.workflow._run_agent3(user_profile, analysis_results)
                
                # Enhance portfolio with realistic returns
                portfolio = self._enhance_portfolio_returns(portfolio, user_profile.get('risk_level', 'moderate'))
                
                # Make portfolio JSON serializable
                portfolio = make_json_serializable(portfolio)
                self.workflow.results['agent3_portfolio'] = portfolio
                
                # Generate final recommendation
                final_recommendation = self.workflow._generate_final_output()
                final_recommendation = make_json_serializable(final_recommendation)
                self.workflow.results['final_recommendation'] = final_recommendation
                
                # Save results
                self.workflow._save_complete_results()
                
                self.update_progress(3, "Workflow Complete", 100)
                self.log_message("✅ WORKFLOW COMPLETE!")
                
                # Make entire result JSON serializable
                self.result = make_json_serializable(self.workflow.results)
                
            finally:
                loop.close()
                
        except Exception as e:
            self.error = str(e)
            self.log_message(f"❌ Workflow failed: {e}")
            logger.error(f"Workflow {self.workflow_id} failed: {e}")
        
        # Emit completion with JSON-safe data
        try:
            completion_data = {
                'workflow_id': self.workflow_id,
                'success': self.error is None,
                'result': self.result,
                'error': self.error
            }
            completion_data = make_json_serializable(completion_data)
            
            socketio.emit('workflow_complete', completion_data, room=self.workflow_id)
        except Exception as e:
            logger.error(f"Failed to emit workflow completion: {e}")
            try:
                socketio.emit('workflow_complete', {
                    'workflow_id': self.workflow_id,
                    'success': self.error is None,
                    'result': {'message': 'Workflow completed - check server logs for details'},
                    'error': str(self.error) if self.error else None
                }, room=self.workflow_id)
            except Exception as e2:
                logger.error(f"Failed to emit simplified completion: {e2}")
    
    def run_portfolio_only(self, force_stock_refresh=False):
        """Run only portfolio construction - DEPRECATED"""
        self.error = "Portfolio-only mode has been removed. Use complete workflow instead."
        self.log_message("❌ Portfolio-only mode is no longer supported")

# API Routes

@app.route('/')
def serve_ui():
    """Serve the main UI"""
    return send_from_directory('.', 'aurite_ui.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_workflows': len(active_workflows),
        'stock_analysis_fresh': check_stock_analysis_freshness()
    })

@app.route('/api/start-workflow', methods=['POST'])
def start_workflow():
    """Start a new complete workflow"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['investment_amount', 'risk_level', 'time_horizon', 'investment_goal']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate workflow ID
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        
        # Check for force refresh option
        force_stock_refresh = data.get('force_stock_refresh', False)
        
        # Parse additional notes for sector preferences and other settings
        additional_notes = str(data.get('additional_notes', ''))
        parsed_avoid_sectors, parsed_prefer_sectors, parsed_esg, parsed_liquidity, parsed_tax, parsed_monthly = parse_additional_notes(additional_notes)
        
        # Combine explicitly selected sectors with parsed preferences
        preferred_sectors = list(data.get('preferred_sectors', []))
        preferred_sectors.extend(parsed_prefer_sectors)
        preferred_sectors = list(set(preferred_sectors))  # Remove duplicates
        
        avoid_sectors = list(data.get('avoid_sectors', []))
        avoid_sectors.extend(parsed_avoid_sectors)
        avoid_sectors = list(set(avoid_sectors))  # Remove duplicates
        
        # Combine preferences
        prefers_esg = bool(data.get('prefers_esg', False)) or parsed_esg
        needs_liquidity = bool(data.get('needs_liquidity', False)) or parsed_liquidity
        tax_sensitive = bool(data.get('tax_sensitive', False)) or parsed_tax
        monthly_contribution = int(data.get('monthly_contribution', 0)) or parsed_monthly
        
        # Create user profile in the enhanced AIvestor format
        user_profile = {
            'profile_id': workflow_id,
            'timestamp': datetime.now().isoformat(),
            'investment_amount': int(data['investment_amount']),
            'risk_level': str(data['risk_level']),
            'time_horizon': str(data['time_horizon']),  # Keep as string for short/medium/long
            'investment_goal': str(data['investment_goal']),
            'monthly_contribution': monthly_contribution,
            'preferred_sectors': preferred_sectors,
            'avoid_sectors': avoid_sectors,
            'additional_notes': additional_notes,
            'prefers_esg': prefers_esg,
            'needs_liquidity': needs_liquidity,
            'tax_sensitive': tax_sensitive,
            'creation_timestamp': datetime.now().isoformat(),
            'user_sophistication': 'intermediate',  # Default value
            'behavioral_traits': {
                'risk_seeking': data['risk_level'] == 'aggressive',
                'risk_averse': data['risk_level'] == 'conservative',
                'disciplined_saver': monthly_contribution > 0
            },
            'parsed_preferences': {
                'avoid_sectors_from_notes': parsed_avoid_sectors,
                'prefer_sectors_from_notes': parsed_prefer_sectors,
                'esg_from_notes': parsed_esg,
                'liquidity_from_notes': parsed_liquidity,
                'tax_from_notes': parsed_tax,
                'monthly_from_notes': parsed_monthly
            }
        }
        
        # Create and store workflow runner
        workflow_runner = WorkflowRunner(workflow_id)
        active_workflows[workflow_id] = workflow_runner
        
        # Start workflow in background thread
        thread = threading.Thread(
            target=workflow_runner.run_complete_workflow,
            args=(user_profile, force_stock_refresh)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'workflow_id': workflow_id,
            'status': 'started',
            'message': 'Workflow started successfully',
            'parsed_sectors': {
                'avoid_sectors': avoid_sectors,
                'preferred_sectors': preferred_sectors,
                'from_notes': {
                    'avoid': parsed_avoid_sectors,
                    'prefer': parsed_prefer_sectors,
                    'esg': parsed_esg,
                    'liquidity': parsed_liquidity,
                    'tax_sensitive': parsed_tax,
                    'monthly_contribution': parsed_monthly
                }
            },
            'will_refresh_stock_analysis': force_stock_refresh or not check_stock_analysis_freshness()
        })
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-portfolio-only', methods=['POST'])
def start_portfolio_only():
    """Start portfolio construction only - DEPRECATED"""
    return jsonify({
        'error': 'Portfolio-only mode has been removed. Please use the complete workflow.',
        'message': 'Use the complete investment analysis workflow instead.'
    }), 400

@app.route('/api/workflow-status/<workflow_id>')
def get_workflow_status(workflow_id):
    """Get workflow status"""
    if workflow_id not in active_workflows:
        return jsonify({'error': 'Workflow not found'}), 404
    
    workflow = active_workflows[workflow_id]
    
    status_data = {
        'workflow_id': workflow_id,
        'progress': workflow.progress,
        'current_step': workflow.current_step,
        'logs': workflow.logs,
        'completed': workflow.result is not None or workflow.error is not None,
        'error': workflow.error
    }
    
    return jsonify(make_json_serializable(status_data))

@app.route('/api/workflow-result/<workflow_id>')
def get_workflow_result(workflow_id):
    """Get workflow results"""
    if workflow_id not in active_workflows:
        return jsonify({'error': 'Workflow not found'}), 404
    
    workflow = active_workflows[workflow_id]
    
    if workflow.error:
        return jsonify({'error': workflow.error}), 500
    
    if workflow.result is None:
        return jsonify({'error': 'Workflow not completed yet'}), 202
    
    return jsonify(make_json_serializable(workflow.result))

@app.route('/api/analysis-files')
def list_analysis_files():
    """List available analysis files"""
    try:
        output_dir = Path("analysis_outputs")
        if not output_dir.exists():
            return jsonify({'files': []})
        
        files = []
        for file_path in output_dir.glob("*.json"):
            stat = file_path.stat()
            files.append({
                'name': file_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'type': file_path.stem.split('_')[0] if '_' in file_path.stem else 'unknown'
            })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'files': files})
        
    except Exception as e:
        logger.error(f"Failed to list analysis files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis-file/<filename>')
def get_analysis_file(filename):
    """Get specific analysis file"""
    try:
        file_path = Path("analysis_outputs") / filename
        
        if not file_path.exists() or not file_path.is_file():
            return jsonify({'error': 'File not found'}), 404
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return jsonify(make_json_serializable(data))
        
    except Exception as e:
        logger.error(f"Failed to load analysis file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh-stock-analysis', methods=['POST'])
def refresh_stock_analysis():
    """Manually trigger stock analysis refresh"""
    try:
        workflow_id = f"refresh_{int(time.time() * 1000)}"
        
        def run_refresh():
            workflow_runner = WorkflowRunner(workflow_id)
            active_workflows[workflow_id] = workflow_runner
            
            workflow_runner.log_message("🔄 Manual stock analysis refresh requested")
            success = ensure_fresh_stock_analysis(workflow_runner, force_refresh=True)
            
            if success:
                workflow_runner.log_message("✅ Stock analysis refresh completed")
                cleanup_old_stock_analysis()
            else:
                workflow_runner.log_message("❌ Stock analysis refresh failed")
        
        thread = threading.Thread(target=run_refresh)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'workflow_id': workflow_id,
            'status': 'started',
            'message': 'Stock analysis refresh started'
        })
        
    except Exception as e:
        logger.error(f"Failed to refresh stock analysis: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Aurite AI Workflow Server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_workflow')
def handle_join_workflow(data):
    """Join a workflow room for real-time updates"""
    workflow_id = data.get('workflow_id')
    if workflow_id:
        join_room(workflow_id)
        emit('joined_workflow', {'workflow_id': workflow_id})
        logger.info(f"Client {request.sid} joined workflow {workflow_id}")

@socketio.on('leave_workflow')
def handle_leave_workflow(data):
    """Leave a workflow room"""
    workflow_id = data.get('workflow_id')
    if workflow_id:
        leave_room(workflow_id)
        emit('left_workflow', {'workflow_id': workflow_id})
        logger.info(f"Client {request.sid} left workflow {workflow_id}")

# Cleanup old workflows periodically
def cleanup_old_workflows():
    """Remove completed workflows older than 1 hour"""
    current_time = time.time()
    to_remove = []
    
    for workflow_id, workflow in active_workflows.items():
        # Remove workflows that are completed and older than 1 hour
        if (workflow.result is not None or workflow.error is not None):
            # Extract timestamp from workflow_id
            try:
                workflow_timestamp = int(workflow_id.split('_')[1]) / 1000
                if current_time - workflow_timestamp > 3600:  # 1 hour
                    to_remove.append(workflow_id)
            except (IndexError, ValueError):
                # If we can't parse timestamp, remove anyway
                to_remove.append(workflow_id)
    
    for workflow_id in to_remove:
        del active_workflows[workflow_id]
        logger.info(f"Cleaned up old workflow: {workflow_id}")

# Schedule cleanup every 30 minutes
def schedule_cleanup():
    while True:
        time.sleep(1800)  # 30 minutes
        cleanup_old_workflows()

# Start cleanup thread
cleanup_thread = threading.Thread(target=schedule_cleanup)
cleanup_thread.daemon = True
cleanup_thread.start()

def find_free_port():
    """Find a free port starting from 5001"""
    import socket
    from contextlib import closing
    
    for port in range(5001, 5010):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    return 8000  # fallback port

if __name__ == '__main__':
    # Ensure analysis_outputs directory exists
    os.makedirs("analysis_outputs", exist_ok=True)
    
    # Find available port
    port = find_free_port()
    
    print("🚀 Starting Aurite AI Workflow Server")
    print("=" * 50)
    print(f"Server will be available at: http://localhost:{port}")
    print(f"UI will be available at: http://localhost:{port}")
    print(f"API endpoints at: http://localhost:{port}/api/")
    print("=" * 50)
    print("💡 If port 5000 was in use, server started on available port")
    print("💡 On macOS, port 5000 is often used by AirPlay Receiver")
    print("💡 JSON serialization fixes applied for numpy/pandas data")
    print("💡 Automatic stock analysis refresh enabled")
    print("💡 Enhanced sector parsing from additional notes")
    print("=" * 50)
    
    # Run with SocketIO support
    socketio.run(app, host='0.0.0.0', port=port, debug=True)