import numpy as np

class SimulationConfig:
    """
    这个类用来存储一次模拟的所有配置信息
    支持 SHG (二次谐波) 和 SFG (和频) 两种非线性过程
    """
    def __init__(self, crystal_name, wavelength, temperature, plane, 
                 process_type='SHG', wavelength2=None, sellmeier_source='默认'):
        self.crystal_name = crystal_name  # 存储晶体名
        self.process_type = process_type  # 'SHG' 或 'SFG'
        self.sellmeier_source = sellmeier_source  # Sellmeier方程来源
        
        # 输入波长1（对SHG和SFG都是第一个输入波长）
        self.wavelength1_nm = wavelength
        self.wavelength1_um = wavelength / 1000.0
        
        # 输入波长2（SHG时等于wavelength，SFG时为第二个输入）
        if wavelength2 is None or process_type == 'SHG':
            self.wavelength2_nm = wavelength
            self.wavelength2_um = wavelength / 1000.0
        else:
            self.wavelength2_nm = wavelength2
            self.wavelength2_um = wavelength2 / 1000.0
        
        # 计算输出波长（统一公式）
        if process_type == 'SHG':
            self.wavelength_out_nm = wavelength / 2
        else:  # SFG
            # 1/λ_out = 1/λ₁ + 1/λ₂
            self.wavelength_out_nm = 1 / (1/self.wavelength1_nm + 1/self.wavelength2_nm)
        self.wavelength_out_um = self.wavelength_out_nm / 1000.0
        
        # 保留旧的接口兼容性
        self.wavelength_nm = wavelength
        self.wavelength_um = wavelength / 1000.0
        
        self.plane = plane                # 存储平面
        self.temperature = temperature    # 存储温度
        self.crystal_db = {
            "BBO":  {"group": "3m",     "d": {"d22": 2.2, "d31": 0.04, "d15":0.04, "d11":0.02} },
            "KDP":  {"group": "4bar2m", "d": {"d36": 0.39, "d14": 0.39}     },
            "DKDP": {"group": "4bar2m", "d": {"d36": 0.37, "d14": 0.37}    },
            "CLBO": {"group": "4bar2m", "d": {"d36": 0.95, "d14": 0.95}  },
            "LBO":  {"group": "mm2",    "d": {"d31": 1.05, "d32": 0.85, "d33": 0.05, "d15":1.05, "d24":0.85}},
            "KTP":  {"group": "mm2",    "d": {"d31": 2.20, "d32": 3.70, "d33": 14.6, "d15": 2.2, "d24": 3.7}}
        }

    def get_indices(self, target_wavelength=None, target_temperature=None):
        """
        获取晶体在指定波长和温度下的折射率
        
        参数:
            target_wavelength (float or array): 目标波长(nm)，若为None则使用配置中的波长
            target_temperature (float or array): 目标温度(°C)，若为None则使用配置中的温度
        
        返回:
            dict: 包含折射率的字典 {'n_x': ..., 'n_y': ..., 'n_z': ...}
                  注意: 如果输入是数组，返回的折射率也是数组
        """

        if target_wavelength is None:
            wavelength = self.wavelength_um
        else:
            wavelength = target_wavelength / 1000.0  # 转换为微米

        if target_temperature is None:
            dtemp = self.temperature - 20.0  # 假设20°C为参考温度
        else:
            dtemp = target_temperature - 20.0  # 温度相对于20°C的偏差

        # 根据sellmeier_source和crystal_name选择对应的方程
        if self.crystal_name == "CLBO":
            if self.sellmeier_source == "福晶":
                return self._get_indices_clbo_fujing(wavelength, dtemp)
            else:  # OXIDE或默认
                return self._get_indices_clbo_oxide(wavelength, dtemp)
        elif self.crystal_name == "LBO":
            if self.sellmeier_source == "福晶":
                return self._get_indices_lbo_fujing(wavelength, dtemp)
            else:  # Thorlabs或默认
                return self._get_indices_lbo_thorlabs(wavelength, dtemp)
        else:
            # 其他晶体使用默认方程
            return self._get_indices_default(wavelength, dtemp)
    
    def _get_indices_clbo_oxide(self, wavelength, dtemp):
        """CLBO的OXIDE来源方程（原默认方程）"""
        n_x = np.sqrt(2.2145 + 0.00890 / (wavelength**2 - 0.02051) - 0.01413 * wavelength**2) - 1.9e-6 * dtemp
        n_y = n_x
        n_z = np.sqrt(2.0588 + 0.00866 / (wavelength**2 - 0.01202) - 0.00607 * wavelength**2) - 0.5e-6 * dtemp
        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}
    
    def _get_indices_clbo_fujing(self, wavelength, dtemp):
        """CLBO的福晶来源方程"""
        n_o = np.sqrt(2.2104 + 0.01018 / (wavelength**2 - 0.01424) - 0.01258 * wavelength**2)
        n_e = np.sqrt(2.0588 + 0.00838 / (wavelength**2 - 0.01363) - 0.00607 * wavelength**2)
        # CLBO是单轴晶体，no对应x和y，ne对应z
        # 温度系数暂用OXIDE的值（福晶未提供）
        n_x = n_o - 1.9e-6 * dtemp
        n_y = n_o - 1.9e-6 * dtemp
        n_z = n_e - 0.5e-6 * dtemp
        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}
    
    def _get_indices_lbo_thorlabs(self, wavelength, dtemp):
        """LBO的Thorlabs来源方程（原默认方程）"""
        n_x = np.sqrt(2.4542 + 0.01125 / (wavelength**2 - 0.01135) - 0.01388 * wavelength**2) + ((dtemp + 29.13e-3 * dtemp**2) * ((-3.76 * wavelength + 2.30) * 1e-6))
        n_y = np.sqrt(2.5390 + 0.01277 / (wavelength**2 - 0.01189) - 0.01849 * wavelength**2 + (4.3025e-5) * wavelength**4 - (2.9131e-5) * wavelength**6) +((dtemp - (32.89e-4) * dtemp**2) * (6.01 * wavelength - 19.40) * 1e-6)
        n_z = np.sqrt(2.5865 + 0.01310 / (wavelength**2 - 0.01223) - 0.01862 * wavelength**2 + (4.5778e-5) * wavelength**4 - 3.2526e-5 * wavelength**6) + ((dtemp - (74.49e-4) * dtemp**2) * (1.50 * wavelength - 9.70)* 1e-6)
        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}
    
    def _get_indices_lbo_fujing(self, wavelength, dtemp):
        """LBO的福晶来源方程"""
        n_x_sq = 2.454140 + 0.011249 / (wavelength**2 - 0.011350) - 0.014591 * wavelength**2 - 6.60e-5 * wavelength**4
        n_y_sq = 2.539070 + 0.012711 / (wavelength**2 - 0.012523) - 0.018540 * wavelength**2 + 2.00e-4 * wavelength**4
        n_z_sq = 2.586179 + 0.013099 / (wavelength**2 - 0.011893) - 0.017968 * wavelength**2 - 2.26e-4 * wavelength**4
        
        n_x = np.sqrt(n_x_sq)
        n_y = np.sqrt(n_y_sq)
        n_z = np.sqrt(n_z_sq)
        
        # 应用温度系数（图3提供的）
        # dnx/dT = -9.3×10⁻⁶
        # dny/dT = -13.6×10⁻⁶
        # dnz/dT = (-6.3-2.1λ)×10⁻⁶，λ单位是μm
        n_x = n_x - 9.3e-6 * dtemp
        n_y = n_y - 13.6e-6 * dtemp
        n_z = n_z - (6.3 + 2.1 * wavelength) * 1e-6 * dtemp
        
        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}
    
    def _get_indices_default(self, wavelength, dtemp):
        """其他晶体的默认Sellmeier方程"""
        if self.crystal_name == "BBO":
            n_x = np.sqrt((0.90291 * wavelength**2) / (wavelength**2 - 0.003926) + (0.83155 * wavelength**2) / (wavelength**2 - 0.018786) + (0.76536 * wavelength**2) / (wavelength**2 - 60.01) + 1) - 16.6e-6 * dtemp
            n_y = n_x
            n_z = np.sqrt((1.151075 * wavelength**2) / (wavelength**2 - 0.007142) + (0.21803 * wavelength**2) / (wavelength**2 - 0.02259) + (0.656 * wavelength**2) / (wavelength**2 - 263) + 1) - 9.3e-6 * dtemp
        
        elif self.crystal_name == "KTP":
            #福晶官网数据
            n_x = np.sqrt(3.0065 + 0.03901 / (wavelength**2 - 0.04251) - 0.01327 * wavelength**2) + 1.1e-5 * dtemp
            n_y = np.sqrt(3.0333 + 0.04154 / (wavelength**2 - 0.04547) - 0.01408 * wavelength**2) + 1.3e-5 * dtemp
            n_z = np.sqrt(3.3134 + 0.05694 / (wavelength**2 - 0.05658) - 0.01682 * wavelength**2) + 1.6e-5 * dtemp    

        elif self.crystal_name == "KDP":
            n_x = np.sqrt(2.259276 + 0.01008956 / (wavelength**2 - 0.012942625) + (13.00522 * wavelength**2) / (wavelength**2 - 400))
            n_y = n_x
            n_z = np.sqrt(2.132668 + 0.008637494 / (wavelength**2 - 0.012281043) + (3.2279924 * wavelength**2) / (wavelength**2 - 400))

        elif self.crystal_name == "DKDP":
            n_x = np.sqrt(1.9575544 + (0.2901391 * wavelength**2) / (wavelength**2 - 0.0281399) - 0.02824391 * wavelength**2 + 0.004977826 * wavelength**4)
            n_y = n_x
            n_z = np.sqrt(1.5057799 + (0.6276034 * wavelength**2) / (wavelength**2 - 0.0131558) - 0.01054063 * wavelength**2 + 0.002243821 * wavelength**4)


        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}

