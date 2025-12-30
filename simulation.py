"""
非线性晶体相位匹配模拟器

计算SHG/SFG的临界角、走离角、接受角/波长/温度、温度匹配等。
核心架构：所有计算基于统一的delta_n函数。

作者：陈泓鑫
"""
import numpy as np
import matplotlib.pyplot as plt
from configuration import SimulationConfig
from scipy.optimize import fsolve
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go

class Solver():
    """非线性晶体相位匹配求解器
    
    核心方法：delta_n(mode, θ, λ, T) - 统一的相位失配计算函数
    所有其他函数都基于delta_n构建
    """
    def __init__(self, config):
        """初始化求解器，加载晶体折射率数据并设置权重系数"""
        # 保存配置参数
        self.cfg = config
        self.crystal_db = config.crystal_db
        
        # 获取输入波长1,2和输出波长的折射率
        self.indices_w1 = self.cfg.get_indices(self.cfg.wavelength1_nm)
        self.indices_w2 = self.cfg.get_indices(self.cfg.wavelength2_nm)
        self.indices_out = self.cfg.get_indices(self.cfg.wavelength_out_nm)

        # 平面配置: (不动轴, cos²轴, sin²轴)
        self.plane_config = {
            "XZ": ('n_y', 'n_z', 'n_x'),
            "YZ": ('n_x', 'n_z', 'n_y'),
            "XY": ('n_z', 'n_x', 'n_y')
        }
        
        # 根据所选平面获取对应的轴信息
        self.key_static, self.key_cos, self.key_sin = self.plane_config[self.cfg.plane]

        # ===== 输入波1的折射率 =====
        self.major_axis_w1 = self.indices_w1[self.key_cos]
        self.minor_axis_w1 = self.indices_w1[self.key_sin]
        self.nw1_o = self.indices_w1[self.key_static]
        self.nw1_e_func = self.ne_func(self.major_axis_w1, self.minor_axis_w1)
        
        # ===== 输入波2的折射率 =====
        self.major_axis_w2 = self.indices_w2[self.key_cos]
        self.minor_axis_w2 = self.indices_w2[self.key_sin]
        self.nw2_o = self.indices_w2[self.key_static]
        self.nw2_e_func = self.ne_func(self.major_axis_w2, self.minor_axis_w2)
        
        # ===== 输出波的折射率 =====
        self.major_axis_out = self.indices_out[self.key_cos]
        self.minor_axis_out = self.indices_out[self.key_sin]
        self.nout_o = self.indices_out[self.key_static]
        self.nout_e_func = self.ne_func(self.major_axis_out, self.minor_axis_out)
        
        # 权重系数: SHG=(0.5, 0.5), SFG=(λ_out/λ₁, λ_out/λ₂)
        if self.cfg.process_type == 'SHG':
            self.weight1 = 0.5
            self.weight2 = 0.5
        else:  # SFG
            self.weight1 = self.cfg.wavelength_out_nm / self.cfg.wavelength1_nm
            self.weight2 = self.cfg.wavelength_out_nm / self.cfg.wavelength2_nm
        

        # 构建模式列表和equations_deltan包装器（内部调用delta_n）
        if self.cfg.process_type == 'SHG':
            λω = f"{self.cfg.wavelength1_nm:.0f}nm"
            λ2ω = f"{self.cfg.wavelength_out_nm:.0f}nm"
            self.mode_names = [
                f"𝐎 ({λω}) + 𝐎 ({λω}) → 𝐄 ({λ2ω}) (Type I)",
                f"𝐄 ({λω}) + 𝐄 ({λω}) → 𝐎 ({λ2ω}) (Type I)",
                f"𝐎 ({λω}) + 𝐄 ({λω}) → 𝐄 ({λ2ω}) (Type II)",
                f"𝐎 ({λω}) + 𝐄 ({λω}) → 𝐎 ({λ2ω}) (Type II)"
            ]
        else:
            λ1 = f"{self.cfg.wavelength1_nm:.0f}nm"
            λ2 = f"{self.cfg.wavelength2_nm:.0f}nm"
            λout = f"{self.cfg.wavelength_out_nm:.0f}nm"
            self.mode_names = [
                f"𝐎 ({λ1}) + 𝐎 ({λ2}) → 𝐄 ({λout}) (Type I)",
                f"𝐄 ({λ1}) + 𝐄 ({λ2}) → 𝐎 ({λout}) (Type I)",
                f"𝐎 ({λ1}) + 𝐄 ({λ2}) → 𝐄 ({λout}) (Type II)",
                f"𝐎 ({λ2}) + 𝐄 ({λ1}) → 𝐄 ({λout}) (Type II)",
                f"𝐎 ({λ1}) + 𝐄 ({λ2}) → 𝐎 ({λout}) (Type II)",
                f"𝐎 ({λ2}) + 𝐄 ({λ1}) → 𝐎 ({λout}) (Type II)"
            ]
        
        # 为向后兼容，保留 equations_deltan 作为 delta_n 的包装器
        # 每个模式都是一个 lambda，内部调用统一的 delta_n 函数
        self.equations_deltan = {
            mode: (lambda m: lambda theta: self.delta_n(m, theta=theta))(mode)
            for mode in self.mode_names
        }

    def ne_func(self, n_cos, n_sin):
        """
        计算单轴晶体中E光(非寻常光)的有效折射率
        
        对于单轴晶体中的角度相关传播，E光的有效折射率由两个主折射率通过椭球方程混合计算。
        这是晶体光学中的基本公式。
        
        参数:
            n_cos (float): 与cos²θ相关联的折射率(通常为ne或n_max)
            n_sin (float): 与sin²θ相关联的折射率(通常为no或n_min)
        
        返回:
            function: 返回一个关于角度θ的函数 n_e(θ)
        
        公式推导 (单轴晶体椭球方程):
            1/n_e²(θ) = cos²(θ)/n_cos² + sin²(θ)/n_sin²
            
            求解得: n_e(θ) = √[ (n_cos² * n_sin²) / (n_cos² * cos²θ + n_sin² * sin²θ) ]
        
        物理意义:
            - 当θ=0°时,n_e = n_cos (沿主轴)
            - 当θ=90°时,n_e = n_sin (垂直主轴)
            - 中间值通过椭球插值计算
        
        应用:
            在非线性光学中，通过改变传播方向(扫描θ)来改变E光的有效折射率，
            从而调整相位匹配条件
        """
        return lambda theta: np.sqrt(
            (n_cos**2 * n_sin**2) / 
            (n_cos**2 * np.cos(theta)**2 + n_sin**2 * np.sin(theta)**2)
        )

    def delta_n(self, mode_name, theta=None, wavelength1=None, wavelength2=None, 
                wavelength_out=None, temperature=None):
        """
        统一的相位失配 Δn 计算函数
        
        这是整个仿真系统的核心函数，Δn 是一个关于 (θ, λ₁, λ₂, λ_out, T) 的多元函数。
        相位匹配条件即 Δn = 0。不同的带宽计算通过固定某些变量、扫描其他变量来实现。
        
        支持两种偏振表示法：
        1. **OE表示法**（角度调谐）: "𝐎 (1064nm) + 𝐄 (1064nm) → 𝐄 (532nm) (Type II)"
           - 𝐎: O光（寻常光），折射率不随角度变化
           - 𝐄: E光（非寻常光），折射率随角度变化，需要提供theta参数
           
        2. **XYZ表示法**（非临界相位匹配/温度调谐）: "𝐗 (1064nm) + 𝐗 (1064nm) → 𝐘 (532nm) (Type I)"
           - 𝐗/𝐘/𝐙: 沿该主轴偏振，直接使用主轴折射率
           - 用于固定传播方向的温度调谐场景
        
        物理意义:
            Δn 表示相位失配程度，对于和频/倍频过程:
            - SHG: Δn = weight1·n_ω1 + weight2·n_ω2 - n_2ω  (权重归一化后)
            - SFG: Δn = (λ_out/λ₁)·n_1 + (λ_out/λ₂)·n_2 - n_out
        
        参数:
            mode_name (str): 相位匹配模式名称
                OE表示法示例: "𝐎 (1064nm) + 𝐎 (1064nm) → 𝐄 (532nm) (Type I)"
                XYZ表示法示例: "𝐗 (1064nm) + 𝐗 (1064nm) → 𝐘 (532nm) (Type I)"
            theta (float or array): 相位匹配角（弧度），None 则使用当前配置值
                注意：XYZ表示法不需要theta参数
            wavelength1 (float or array): 输入光1波长（nm），None 则使用当前配置值
            wavelength2 (float or array): 输入光2波长（nm），None 则使用当前配置值
            wavelength_out (float or array): 输出光波长（nm），None 则使用当前配置值
            temperature (float or array): 温度（°C），None 则使用当前配置值
        
        返回:
            float or array: 相位失配 Δn 值
            
        使用示例:
            # OE表示法：角度调谐
            theta_range = np.linspace(0, np.pi/2, 1000)
            delta_n_values = [solver.delta_n("𝐎 + 𝐎 → 𝐄 (Type I)", theta=t) for t in theta_range]
            
            # OE表示法：波长带宽
            wl_range = np.linspace(1000, 1100, 1000)
            delta_n_values = [solver.delta_n("𝐎 + 𝐎 → 𝐄 (Type I)", theta=θ_c, wavelength1=w) for w in wl_range]
            
            # XYZ表示法：温度调谐（非临界相位匹配）
            temp_range = np.linspace(20, 200, 1000)
            delta_n_values = [solver.delta_n("𝐗 + 𝐗 → 𝐘 (Type I)", temperature=t) for t in temp_range]
        """
        # 参数默认值填充
        wl1 = wavelength1 if wavelength1 is not None else self.cfg.wavelength1_nm
        wl2 = wavelength2 if wavelength2 is not None else self.cfg.wavelength2_nm
        wl_out = wavelength_out if wavelength_out is not None else self.cfg.wavelength_out_nm
        temp = temperature if temperature is not None else self.cfg.temperature
        
        # 获取折射率
        indices_w1 = self.cfg.get_indices(target_wavelength=wl1, target_temperature=temp)
        indices_w2 = self.cfg.get_indices(target_wavelength=wl2, target_temperature=temp)
        indices_out = self.cfg.get_indices(target_wavelength=wl_out, target_temperature=temp)
        
        nw1_o = indices_w1[self.key_static]
        nw1_e_func = self.ne_func(indices_w1[self.key_cos], indices_w1[self.key_sin])
        nw2_o = indices_w2[self.key_static]
        nw2_e_func = self.ne_func(indices_w2[self.key_cos], indices_w2[self.key_sin])
        nout_o = indices_out[self.key_static]
        nout_e_func = self.ne_func(indices_out[self.key_cos], indices_out[self.key_sin])
        
        # 解析模式名称，支持OE和XYZ两种表示法
        parts = mode_name.split('→')
        if len(parts) != 2:
            raise ValueError(f"模式名称格式错误: {mode_name}")
        
        input_part = parts[0].strip()
        output_part = parts[1].strip()
        is_xyz_notation = any(c in mode_name for c in ['𝐗', '𝐘', '𝐙'])
        
        if is_xyz_notation:
            # XYZ表示法：直接使用主轴折射率（非临界相位匹配）
            def extract_xyz_pol(text):
                for pol in ['𝐗', '𝐘', '𝐙']:
                    if pol in text:
                        return pol
                return None
            
            input_beams = input_part.split('+')
            pol1 = extract_xyz_pol(input_beams[0])
            pol2 = extract_xyz_pol(input_beams[1]) if len(input_beams) > 1 else pol1
            pol_out = extract_xyz_pol(output_part)
            
            xyz_to_key = {'𝐗': 'n_x', '𝐘': 'n_y', '𝐙': 'n_z'}
            n1 = indices_w1[xyz_to_key[pol1]]
            n2 = indices_w2[xyz_to_key[pol2]]
            n_out = indices_out[xyz_to_key[pol_out]]
            
        else:
            # OE表示法：根据角度计算E光折射率
            # 需要识别模式字符串中波长的顺序，匹配到正确的配置参数
            import re
            
            # 提取所有波长信息 (格式: "1064nm")
            wavelengths_in_mode = re.findall(r'(\d+)nm', mode_name)
            if len(wavelengths_in_mode) < 3:
                raise ValueError(f"无法从模式字符串中提取波长信息: {mode_name}")
            
            wl_beam1_str = float(wavelengths_in_mode[0])  # 第一束光波长（模式字符串中的）
            wl_beam2_str = float(wavelengths_in_mode[1])  # 第二束光波长（模式字符串中的）
            
            # 判断波长顺序：比较模式字符串中的波长与配置文件中的波长
            # 如果第一个波长接近wavelength1，说明顺序一致；否则是交换的
            tolerance = 1.0  # 容差1nm
            if abs(wl_beam1_str - wl1) < tolerance:
                # 顺序一致：beam1用wl1, beam2用wl2
                indices_beam1 = indices_w1
                indices_beam2 = indices_w2
                actual_wl1 = wl1
                actual_wl2 = wl2
            else:
                # 顺序相反：beam1用wl2, beam2用wl1
                indices_beam1 = indices_w2
                indices_beam2 = indices_w1
                actual_wl1 = wl2
                actual_wl2 = wl1
            
            indices_output = indices_out
            actual_wl_out = wl_out
            
            # 提取偏振顺序
            input_pols = []
            if '𝐎' in input_part:
                input_pols.append(('𝐎', input_part.index('𝐎')))
            if '𝐄' in input_part:
                input_pols.append(('𝐄', input_part.index('𝐄')))
            input_pols.sort(key=lambda x: x[1])
            pol1 = input_pols[0][0]  # 第一束光的偏振
            pol2 = input_pols[1][0] if len(input_pols) > 1 else input_pols[0][0]  # 第二束光的偏振
            pol_out = '𝐄' if '𝐄' in output_part.split('(')[0] else '𝐎'
            
            if theta is None and (pol1 == '𝐄' or pol2 == '𝐄' or pol_out == '𝐄'):
                raise ValueError("计算E光时必须提供theta参数")
            
            # 第一束光的折射率
            if pol1 == '𝐎':
                n1 = indices_beam1[self.key_static]
            else:
                ne1_func = self.ne_func(indices_beam1[self.key_cos], indices_beam1[self.key_sin])
                n1 = ne1_func(theta)
            
            # 第二束光的折射率
            if pol2 == '𝐎':
                n2 = indices_beam2[self.key_static]
            else:
                ne2_func = self.ne_func(indices_beam2[self.key_cos], indices_beam2[self.key_sin])
                n2 = ne2_func(theta)
            
            # 输出光的折射率
            if pol_out == '𝐎':
                n_out = indices_output[self.key_static]
            else:
                ne_out_func = self.ne_func(indices_output[self.key_cos], indices_output[self.key_sin])
                n_out = ne_out_func(theta)
            
            # 根据实际波长计算正确的权重
            if self.cfg.process_type == 'SHG':
                w1 = 0.5
                w2 = 0.5
            else:  # SFG: 权重 = λ_out / λ_beam
                w1 = actual_wl_out / actual_wl1
                w2 = actual_wl_out / actual_wl2
        
        # 计算Δn（使用正确的权重）
        if is_xyz_notation:
            # XYZ模式使用预设的权重
            delta_n_value = self.weight1 * n1 + self.weight2 * n2 - n_out
        else:
            # OE模式使用根据实际波长计算的权重
            delta_n_value = w1 * n1 + w2 * n2 - n_out
        
        return delta_n_value

    def criticalangle(self):
        """计算相位匹配的临界角度，对所有模式求解Δn=0"""
        
        # ===== 内部求解函数: robust_solve =====
        def robust_solve(equation_func, guess=np.pi/4):
            """
            数值求解器：尝试找到方程的根，失败或无解时返回 np.nan
            
            参数:
                equation_func: 目标方程 f(θ)，当 f(θ)=0 时满足相位匹配
                guess: 初始猜测值,默认45°(π/4弧度)，这是比较合理的起点
            
            返回:
                float: 求解得到的角度(弧度)，或 np.nan(无解)
            
            鲁棒性保证:
                1. fsolve 返回信息 ier=1 表示成功收敛
                2. 解必须在物理范围 [0°, 90°] = [0, π/2] 内
                3. 将解代回原方程验证，残差 |f(θ_solution)| < 1e-4
                4. 严格检验防止伪收敛和数值不稳定
            
            参数说明:
                - full_output=1: 让 fsolve 返回详细信息，包括收敛标志 ier
                - ier=1: 收敛成功
                - ier≠1: 求解失败或不收敛
            """
            # 调用 scipy 的 fsolve 非线性方程求解器
            # full_output=1 可以获得收敛信息
            root, _, ier, _ = fsolve(equation_func, guess, full_output=1)      
                     
            if ier == 1:
                theta_res = root[0]
                # ===== 双重检查 =====
                # 检验 1: 解必须在物理范围内 (0° 到 90°)
                if 0 <= theta_res <= np.pi/2:
                    # 检验 2: 把解代回方程，计算残差
                    # 如果残差太大，说明是伪收敛，应该舍弃
                    residual = abs(equation_func(theta_res))
                    if residual < 1e-4:
                        return theta_res
            
            # 如果求解失败或未通过检验，返回 NaN
            return np.nan

        # 遍历所有模式，求解Δn=0的角度
        theta_critical_dict_results = {}
        for mode_name, eq_func in self.equations_deltan.items():
            theta_val = robust_solve(eq_func, guess=np.pi/4)
            theta_deg = np.rad2deg(theta_val) if not np.isnan(theta_val) else np.nan
            theta_critical_dict_results[mode_name] = theta_deg
        return theta_critical_dict_results

    def walkoff_angle(self, theta_critical_dict, phi):
        """计算走离角: ρ = θ - arctan(a²/b² * tanθ)，只有E光有走离角"""

        walkoff_angle_results = {}

        for mode_name, theta_deg in theta_critical_dict.items():
            if np.isnan(theta_deg):
                walkoff_angle_results[mode_name] = np.nan
            else:
                theta_rad = np.deg2rad(theta_deg)
                
                # 解析模式名称提取偏振信息
                parts = mode_name.split('→')
                if len(parts) != 2:
                    walkoff_angle_results[mode_name] = "格式错误"
                    continue
                
                input_part, output_part = parts[0].strip(), parts[1].strip()
                
                input_pols = []
                if '𝐎' in input_part:
                    input_pols.append(('𝐎', input_part.index('𝐎')))
                if '𝐄' in input_part:
                    input_pols.append(('𝐄', input_part.index('𝐄')))
                input_pols.sort(key=lambda x: x[1])
                pol1, pol2 = input_pols[0][0], input_pols[1][0] if len(input_pols) > 1 else input_pols[0][0]
                pol_out = '𝐄' if '𝐄' in output_part.split('(')[0] else '𝐎'
                
                def calc_walkoff(pol, wavelength_nm):
                    """计算指定偏振和波长的走离角"""
                    if pol == '𝐎':
                        return 0.0, 0.0
                    
                    indices = self.cfg.get_indices(wavelength_nm)
                    n_x, n_y, n_z = indices['n_x'], indices['n_y'], indices['n_z']
                    
                    plane = self.cfg.plane
                    if plane == "XY":
                        a, b = n_x, n_y
                    elif plane == "XZ":
                        a, b = n_z, n_x
                    else:
                        a, b = n_z, n_y
                    
                    tan_theta_normal = (a**2 / b**2) * np.tan(theta_rad)
                    theta_normal_rad = np.arctan(tan_theta_normal)
                    rho_rad = theta_rad - theta_normal_rad
                    rho_deg = np.rad2deg(rho_rad)
                    rho_mrad = rho_rad * 1e3
                    
                    return rho_deg, rho_mrad
                
                wavelength1 = self.cfg.wavelength1_nm
                wavelength2 = self.cfg.wavelength2_nm if self.cfg.process_type == 'SFG' else wavelength1
                wavelength_out = self.cfg.wavelength_out_nm
                
                rho1_deg, rho1_mrad = calc_walkoff(pol1, wavelength1)
                rho2_deg, rho2_mrad = calc_walkoff(pol2, wavelength2)
                rho_out_deg, rho_out_mrad = calc_walkoff(pol_out, wavelength_out)
                
                def format_walkoff(pol, deg, mrad):
                    if pol == '𝐎':
                        return f"{pol}  (0°)"
                    else:
                        return f"{pol}  ({deg:.4f}° / {mrad:.4f} mrad)"
                
                result_str = " {} | {} | {}".format(
                    format_walkoff(pol1, rho1_deg, rho1_mrad),
                    format_walkoff(pol2, rho2_deg, rho2_mrad),
                    format_walkoff(pol_out, rho_out_deg, rho_out_mrad)
                )
                
                walkoff_angle_results[mode_name] = result_str
                
        return walkoff_angle_results

    def d_eff(self, theta_critical_dict, selected_phi=None):
        """计算有效非线性系数 d_eff，根据晶体点群对称性和相位匹配几何构型计算
        
        核心逻辑：
        1. 从模式字符串识别匹配类型：OOE, EEO, OEE, OEO
        2. 根据平面确定theta和phi角
        3. 用匹配类型和角度参数计算d_eff
        """
        crystal_info = self.crystal_db[self.cfg.crystal_name]
        if not crystal_info:
            return {}
        
        d_tensor = crystal_info["d"]
        d_eff_dict = {}
        
        # 步骤1: 识别每个模式的匹配类型
        def get_mode_type(mode_name):
            """提取模式类型: 'OOE', 'EEO', 'OEE', 'OEO'"""
            if '→' not in mode_name:
                return None
            
            parts = mode_name.split('→')
            input_part = parts[0].strip()
            output_part = parts[1].strip()
            
            # 提取偏振符号
            input_pols = [c for c in input_part if c in ['𝐎', '𝐄']]
            output_pol = next((c for c in output_part if c in ['𝐎', '𝐄']), None)
            
            if len(input_pols) < 2 or output_pol is None:
                return None
            
            # 返回三字符模式类型（不区分顺序，OE和EO都算OE）
            pol1, pol2 = input_pols[0], input_pols[1]
            if pol1 == pol2:
                # Type I
                return f"{pol1}{pol2}{output_pol}".replace('𝐎', 'O').replace('𝐄', 'E')
            else:
                # Type II (统一为OE)
                return f"OE{output_pol}".replace('𝐎', 'O').replace('𝐄', 'E')
        
        # 步骤2: 确定每个模式的theta和phi角
        for mode_name, critical_angle_deg in theta_critical_dict.items():
            mode_type = get_mode_type(mode_name)
            if mode_type is None:
                d_eff_dict[mode_name] = 0.0
                continue
            
            # 根据平面确定theta和phi
            if self.cfg.plane == "XY":
                # XY平面：phi是相位匹配角，theta固定90°
                theta_rad = np.deg2rad(90.0)
                phi_rad = np.deg2rad(critical_angle_deg)
            elif self.cfg.plane == "XZ":
                # XZ平面：theta是相位匹配角，phi固定0°
                theta_rad = np.deg2rad(critical_angle_deg)
                phi_rad = np.deg2rad(selected_phi if selected_phi is not None else 0.0)
            elif self.cfg.plane == "YZ":
                # YZ平面：theta是相位匹配角，phi固定90°
                theta_rad = np.deg2rad(critical_angle_deg)
                phi_rad = np.deg2rad(selected_phi if selected_phi is not None else 90.0)
            else:
                d_eff_dict[mode_name] = 0.0
                continue
            
            # 步骤3: 根据晶体点群和模式类型计算d_eff
            d_value = 0.0
            
            if crystal_info["group"] == "4bar2m":  # BBO类晶体
                d36 = d_tensor.get('d36', 0)
                if mode_type == "OOE":
                    d_value = d36 * np.sin(theta_rad) * np.sin(2*phi_rad)
                elif mode_type == "EEO":
                    d_value = d36 * np.sin(2*theta_rad) * np.cos(2*phi_rad)
                elif mode_type == "OEE":
                    d_value = d36 * np.sin(2*theta_rad) * np.cos(2*phi_rad)
                elif mode_type == "OEO":
                    d_value = d36 * np.sin(theta_rad) * np.sin(2*phi_rad)
            
            elif crystal_info["group"] == "3m":  # 三方晶系
                d31 = d_tensor.get('d31', 0)
                d11 = d_tensor.get('d11', 0)
                d22 = d_tensor.get('d22', 0)
                d15 = d_tensor.get('d15', 0)
                
                if mode_type == "OOE":
                    d_value = d31 * np.sin(theta_rad) + (d11*np.cos(3*phi_rad) - d22*np.sin(3*phi_rad)) * np.cos(theta_rad)
                elif mode_type == "EEO":
                    d_value = d31 * np.sin(theta_rad) + (d22*np.sin(3*phi_rad) - d11*np.cos(3*phi_rad)) * np.cos(theta_rad)
                elif mode_type == "OEE":
                    d_value = (d11*np.sin(3*phi_rad) + d22*np.cos(3*phi_rad)) * np.cos(theta_rad)**2
                elif mode_type == "OEO":
                    d_value = d15 * np.sin(theta_rad) + (d11*np.cos(3*phi_rad) - d22*np.sin(3*phi_rad)) * np.cos(theta_rad)
            
            elif crystal_info["group"] == "mm2":  # LBO, KTP类晶体
                d31 = d_tensor.get('d31', 0)
                d32 = d_tensor.get('d32', 0)
                d33 = d_tensor.get('d33', 0)
                
                if self.cfg.plane == "XY":
                    if mode_type == "OOE":
                        d_value = d31 * np.cos(phi_rad)**2 + d32 * np.sin(phi_rad)**2
                    elif mode_type == "EEO":
                        d_value = d33
                    # Type II在XY平面为0
                elif self.cfg.plane == "YZ":
                    if mode_type == "OOE":
                        d_value = d31 * np.cos(theta_rad)
                    elif mode_type in ["OEE", "OEO"]:
                        d_value = d31 * np.sin(theta_rad)
                elif self.cfg.plane == "XZ":
                    if mode_type == "OOE":
                        d_value = d32 * np.cos(theta_rad)
                    elif mode_type in ["OEE", "OEO"]:
                        d_value = d32 * np.sin(theta_rad)
            
            d_eff_dict[mode_name] = abs(d_value)
        
        return d_eff_dict

    def acceptance_angle(self, theta_critical_dict, target_mode, step=1000, res=0.1):
        """计算相位匹配接受角：扫描临界角附近的角度范围，计算转换效率并找FWHM"""
        # ===== 构建角度扫描数组 =====
        # 以临界角为中心，前后各扫描 step 个点
        # 单位变换: mrad × 1e-3 = rad
        theta_axis = np.deg2rad(theta_critical_dict[target_mode]) + np.arange(-step, step) * res * 1e-3 
       
        # ===== 使用统一的delta_n函数计算相位失配 =====
        # 对每个角度计算Δn，使用当前配置的波长和温度
        delta_n_array = np.array([self.delta_n(target_mode, theta=t) for t in theta_axis])
        
        # 计算Δk：Δk = 2π/λ_out × Δn
        delta_k_angle = (np.pi * 2 / self.cfg.wavelength_out_um) * delta_n_array
        
        # 转换效率: η(Δk) = sinc²(Δk × L/2)
        # sinc(x) = sin(x)/x
        efficiency_angle = (np.sinc(delta_k_angle * 1e4 / (2 * np.pi)))**2

        # ===== 绘制接受角曲线 =====
        fig,ax = plt.subplots(figsize=(10, 6))
        ax.plot(theta_axis * 1000, efficiency_angle, 'r-', linewidth=1.5)
        ax.set_xlabel('Angle Deviation / mrad', fontsize=12)  # X轴: 角度偏差(毫弧度)
        # 根据过程类型设置纵轴标题
        ylabel = 'SHG Efficiency' if self.cfg.process_type == 'SHG' else 'SFG Efficiency'
        ax.set_ylabel(ylabel, fontsize=12)
        # 替换Unicode粗体字符为普通字符以便在图表中正确显示
        display_mode = target_mode.replace('𝐎', 'O').replace('𝐄', 'E')
        ax.set_title(f'Acceptance Angle Curve for {self.cfg.crystal_name}\n({display_mode})', fontsize=14)
        ax.grid(True, alpha=0.3)

        # ===== 计算接受角(FWHM, 半高全宽) =====
        # FWHM 定义: 效率降到最大值50%时的角度范围
        half_max = 0.5
        
        # 找出所有效率≥50%的点
        indices_above_half = np.where(efficiency_angle >= half_max)[0]
        
        acceptance_angle = np.nan  # 默认值
        acceptance_angle_deg = np.nan  # 默认值
        if len(indices_above_half) > 0:
            # 最小角度对应的索引(左边界)
            lower_index = indices_above_half[0]
            # 最大角度对应的索引(右边界)
            upper_index = indices_above_half[-1]
            
            # 计算接受角(毫弧度)
            acceptance_angle = (theta_axis[upper_index] - theta_axis[lower_index]) * 1000
            
            # 转换为度数便于理解
            acceptance_angle_deg = np.rad2deg(theta_axis[upper_index] - theta_axis[lower_index])



        return fig, acceptance_angle, acceptance_angle_deg

    def acceptance_wavelength(self, theta_critical_dict, target_mode, step, res):
        """
        计算相位匹配接受波长（波长带宽），扫描基频波长附近的范围，计算转换效率并找FWHM
        
        SFG处理策略：假设λ₂是λ₁的高次谐波，当λ₁偏移时λ₂按相同比例同步偏移（ratio = λ₂/λ₁）
        这符合实际应用：激光器波长漂移时基频和谐波光同步变化
        """
        
        wavelength1_axis = self.cfg.wavelength1_nm + np.arange(-step, step) * res 

        if self.cfg.process_type == 'SHG':
            wavelength2_axis = wavelength1_axis
            wavelength_out_axis = wavelength1_axis / 2
            wavelength_ratio = 1.0
        else:
            wavelength_ratio = self.cfg.wavelength2_nm / self.cfg.wavelength1_nm
            wavelength2_axis = wavelength_ratio * wavelength1_axis
            wavelength_out_axis = 1 / (1/wavelength1_axis + 1/wavelength2_axis)

        tem_theta = np.deg2rad(theta_critical_dict[target_mode])
        
        if self.cfg.process_type == 'SHG':
            delta_n_array = np.array([
                self.delta_n(target_mode, theta=tem_theta, 
                            wavelength1=wl, wavelength2=wl,
                            wavelength_out=wl/2)
                for wl in wavelength1_axis
            ])
        else:
            delta_n_array = np.array([
                self.delta_n(target_mode, theta=tem_theta,
                            wavelength1=wl1, wavelength2=wl2,
                            wavelength_out=wl_out)
                for wl1, wl2, wl_out in zip(wavelength1_axis, wavelength2_axis, wavelength_out_axis)
            ])
        
        delta_k_wavelength = (np.pi * 2 / self.cfg.wavelength_out_um) * delta_n_array
        efficiency_wavelength = (np.sinc(delta_k_wavelength * 1e4 / (2 * np.pi)))**2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelength1_axis, efficiency_wavelength, 'g-', linewidth=1.5)
        
        # 根据过程类型设置标签
        if self.cfg.process_type == 'SHG':
            ax.set_xlabel('Fundamental Wavelength Deviation / nm', fontsize=12)
            ax.set_ylabel('SHG Efficiency', fontsize=12)
        else:
            ax.set_xlabel('Fundamental Wavelength Deviation / nm', fontsize=12)
            ax.set_ylabel('SFG Efficiency', fontsize=12)
            # 添加说明文字（使用英文避免字体问题）
            fig.text(0.5, -0.02, 'Note: Wavelength deviations of both beams are proportionally synchronized.', 
                    ha='center', fontsize=10, style='italic', color='gray')
        
        # 替换Unicode粗体字符为普通字符以便在图表中正确显示
        display_mode = target_mode.replace('𝐎', 'O').replace('𝐄', 'E').replace('𝐗', 'X').replace('𝐘', 'Y').replace('𝐙', 'Z')
        ax.set_title(f'Acceptance Wavelength Curve for {self.cfg.crystal_name}\n({display_mode})', fontsize=14)
        ax.grid(True, alpha=0.3)
    
        half_max = 0.5  
        indices_above_half = np.where(efficiency_wavelength >= half_max)[0]
        
        acceptance_wavelength = np.nan
        acceptance_bandwidth = np.nan
        if len(indices_above_half) > 0:
            lower_index = indices_above_half[0]
            upper_index = indices_above_half[-1]
            acceptance_wavelength = (wavelength1_axis[upper_index] - wavelength1_axis[lower_index])
            acceptance_bandwidth = 299792458 / (self.cfg.wavelength1_nm**2) * acceptance_wavelength 

        return fig, acceptance_wavelength, acceptance_bandwidth

    def acceptance_temperature(self, theta_critical_dict ,target_mode, step, res):
        """计算相位匹配接受温度：扫描临界温度附近的范围，计算转换效率并找FWHM"""
        
        temperature_axis = self.cfg.temperature + np.arange(-step, step) * res 

        tem_theta = np.deg2rad(theta_critical_dict[target_mode])
        
        delta_n_array = np.array([
            self.delta_n(target_mode, theta=tem_theta, temperature=temp)
            for temp in temperature_axis
        ])
        
        delta_k_temperature = (np.pi * 2 / self.cfg.wavelength_out_um) * delta_n_array
        efficiency_temperature = (np.sinc(delta_k_temperature * 1e4 / (2 * np.pi)))**2

        # ===== 绘制接受温度曲线 =====
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temperature_axis, efficiency_temperature, 'b-', linewidth=1.5)
        ax.set_xlabel('Temperature Deviation / °C', fontsize=12)  # X轴: 温度偏差(°C)
        # 根据过程类型设置纵轴标题
        ylabel = 'SHG Efficiency' if self.cfg.process_type == 'SHG' else 'SFG Efficiency'
        ax.set_ylabel(ylabel, fontsize=12)
        # 替换Unicode粗体字符为普通字符以便在图表中正确显示
        display_mode = target_mode.replace('𝐎', 'O').replace('𝐄', 'E').replace('𝐗', 'X').replace('𝐘', 'Y').replace('𝐙', 'Z')
        ax.set_title(f'Acceptance Temperature Curve for {self.cfg.crystal_name}\n({display_mode})', fontsize=14)
        ax.grid(True, alpha=0.3) 
    
        # ===== 计算接受温度(FWHM, 半高全宽) =====
        # FWHM: 效率下降到最大值50%时的温度范围
        half_max = 0.5  
        indices_above_half = np.where(efficiency_temperature >= half_max)[0]
        
        acceptance_temperature = np.nan  # 默认值
        if len(indices_above_half) > 0:
            lower_index = indices_above_half[0]
            upper_index = indices_above_half[-1]
            
            acceptance_temperature = (temperature_axis[upper_index] - temperature_axis[lower_index])
            print(f"\n接受温度(Acceptance Temperature (FWHM)): {acceptance_temperature:.4f} K·cm")
        else:
            print("No points found above half maximum efficiency.")

        return fig, acceptance_temperature

    def temperature_phase_matching(self, target_mode, temperature_range=(20, 200), temp_step=0.1, fixed_axis='Z'):
        """温度相位匹配计算：在固定传播轴下扫描温度，找到实现Δn=0的温度点"""
        
        temp_min, temp_max = temperature_range
        temperature_axis = np.arange(temp_min, temp_max + temp_step, temp_step)
        
        phase_mismatch = np.array([
            self.delta_n(target_mode, temperature=temp)
            for temp in temperature_axis
        ])
        
        matching_temperatures = []
        tolerance = 1e-5
        
        for i in range(len(phase_mismatch) - 1):
            if phase_mismatch[i] * phase_mismatch[i + 1] <= 0:
                if abs(phase_mismatch[i+1] - phase_mismatch[i]) > 1e-10:
                    t_exact = temperature_axis[i] - phase_mismatch[i] * (temperature_axis[i+1] - temperature_axis[i]) / (phase_mismatch[i+1] - phase_mismatch[i])
                    matching_temperatures.append(t_exact)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temperature_axis, phase_mismatch, 'b-', linewidth=1.5, label='Phase Mismatch Δn')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Phase Matching Condition')
        
        if matching_temperatures:
            for temp in matching_temperatures:
                ax.axvline(x=temp, color='g', linestyle=':', alpha=0.8)
                ax.text(temp, 0, f'{temp:.1f}°C', rotation=90, 
                       verticalalignment='bottom', horizontalalignment='right')
        
        ax.set_xlabel('Temperature / °C', fontsize=12)
        ax.set_ylabel('Phase Mismatch Δn', fontsize=12)
        # 替换Unicode粗体字符为普通字符以便在图表中正确显示
        display_mode = target_mode.replace('𝐎', 'O').replace('𝐄', 'E').replace('𝐗', 'X').replace('𝐘', 'Y').replace('𝐙', 'Z')
        ax.set_title(f'Temperature Phase Matching for {self.cfg.crystal_name} ({display_mode})\n'
                    f'Fixed axis: {fixed_axis}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        result = {
            'matching_temperatures': matching_temperatures,
            'temperature_axis': temperature_axis,
            'phase_mismatch': phase_mismatch,
            'fixed_axis': fixed_axis,
            'fig': fig,
            'min_phase_mismatch': phase_mismatch.min(),
            'max_phase_mismatch': phase_mismatch.max(),
            'closest_temp': temperature_axis[np.argmin(np.abs(phase_mismatch))],
            'closest_pm': phase_mismatch[np.argmin(np.abs(phase_mismatch))]
        }
    
        return result