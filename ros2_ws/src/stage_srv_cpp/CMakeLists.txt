cmake_minimum_required(VERSION 3.8)
project(stage_srv_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
# 1. 引入生成器和依赖包
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED) 

# 2. 声明接口生成
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StageInference.srv"
  DEPENDENCIES sensor_msgs # 非常重要：因为srv中使用了Image类型
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()