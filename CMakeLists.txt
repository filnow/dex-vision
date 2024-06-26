cmake_minimum_required(VERSION 3.5)

project(dex-vision VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BASE_DIR "$ENV{BUILD_DIR}")

set(OpenCV_DIR "${BASE_DIR}/build")
set(OpenVINO_DIR "${BASE_DIR}/openvino/build")

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED Runtime ONNX)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/src/include"
)

set(PROJECT_SOURCES
        src/main.cpp
        src/mainwindow.cpp
        src/include/mainwindow.h
        src/mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(dex-vision
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        src/include/fastsam.h src/fastsam.cpp
        src/include/customlabel.h src/customlabel.cpp
        src/include/depth.h
        src/include/ovload.h src/ovload.cpp

        resources/resources.qrc
        src/include/inpaint.h src/inpaint.cpp
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET dex-vision APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(dex-vision SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(dex-vision
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(dex-vision PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS} openvino::runtime)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.dex-vision)
endif()
set_target_properties(dex-vision PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS dex-vision
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(dex-vision)
endif()
