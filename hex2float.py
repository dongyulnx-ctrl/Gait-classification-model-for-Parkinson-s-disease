import struct

# HEX字符串
hex_data = "0x428481bd84e81e405371f3bd96f67ebff84ac73d22c7e2bb7f803442eb0b4c440d22294cbdd6f71c401d7bcabdc5f47ebf66a2c43db32bceba7f803442eb0b4c440d88d7a8bd2e831f4082c3d9bd6a5a7ebf24e0c73d7e46bfbb7f803442eb0b4c440dd3bf5fbd96ce1e40ff0c14bea8567ebf478ec63d789e89bb7f803442eb0b4c440d9c4157bda9b21d40582ac4bd46117fbf6db7c63d3e7339bb7f803442eb0b4c440d21abbbbd08771e40bb6fe1bdd85c7ebf1d73c93dc79641bb7f803442eb0b4c"

# 移除开头的0x
hex_data = hex_data[2:]

# 将HEX数据解码成bytes
bytes_data = bytes.fromhex(hex_data)

# 使用struct解析为32位浮点数
# i 代表32位整数，f 代表32位浮点数。
# 此处assume每个浮点数占4 bytes，因此每次增加4 bytes进行解析
floats = [struct.unpack('f', bytes_data[i:i+4])[0] for i in range(0, len(bytes_data), 4)]

print(floats)