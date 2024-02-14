mixamo_structure = {
    "Hips": {
        "right_hip": {},
        "left_hip": {},
        "Spine": {
            "Chest": {
                "Neck": {
                    "Head": {
                        "nose": {}
                    }
                },
                "LeftShoulder": {
                    "LeftArm": {
                        "left_elbow": {
                            "LeftForeArm": {
                                "LeftHand": {
                                    "left_wrist": {} # Mixamo에는 직접적인 손목 조인트가 없을 수 있음
                                }
                            }
                        }
                    }
                },
                "RightShoulder": {
                    "RightArm": {
                        "right_elbow": {
                            "RightForeArm": {
                                "RightHand": {
                                    "right_wrist": {} # Mixamo에는 직접적인 손목 조인트가 없을 수 있음
                                }
                            }
                        }
                    }
                }
            }
        },
        "LeftUpLeg": {
            "left_hip": {
                "LeftLeg": {
                    "left_knee": {
                        "LeftFoot": {
                            "LeftToeBase": {}, # Mixamo에서는 발가락 조인트를 명시적으로 사용하지 않을 수 있음
                            "left_ankle": {} # Mixamo에는 발목에 해당하는 조인트가 발 또는 발가락 근처에 위치할 수 있음
                        }
                    }
                }
            }
        },
        "RightUpLeg": {
            "right_hip": {
                "RightLeg": {
                    "right_knee": {
                        "RightFoot": {
                            "RightToeBase": {}, # Mixamo에서는 발가락 조인트를 명시적으로 사용하지 않을 수 있음
                            "right_ankle": {} # Mixamo에는 발목에 해당하는 조인트가 발 또는 발가락 근처에 위치할 수 있음
                        }
                    }
                }
            }
        }
    }
}

# 이 구조는 Mixamo 캐릭터의 리깅 구조를 따르며, 
# 실제 사용 시에는 각 조인트의 실제
