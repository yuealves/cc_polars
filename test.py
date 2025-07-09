import sys
sys.path.append('./release')
sys.path.append('./debug')
import my_module

a = 12
b = 18
print(f"The GCD of {a} and {b} is {my_module.gcd(a, b)}")
print(f"The LCM of {a} and {b} is {my_module.lcm(a, b)}")
