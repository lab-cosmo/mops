add_subdirectory(external)

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} mops catch)

    add_test(
        NAME ${_name_}
        COMMAND $<TARGET_FILE:${_name_}>
    )
endforeach()
