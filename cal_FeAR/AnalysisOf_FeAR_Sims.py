import Agent



def get_mdr_string(MdR4Agents, return_names=False):
    if return_names:

        mdr_name_list = []
        action_names, _ = Agent.DefineActions()
        for agent, mdr in MdR4Agents:
            mdr_name = action_names[mdr]
            if mdr_name == 'Stay':
                mdr_name_list.append('S0')
            else:
                # First Letter and Last Number (of steps)
                mdr_name_list.append(mdr_name[0] + mdr_name[-1])
        mdr_string_names = '-'.join(mdr_name_list)
        return mdr_string_names

    else:

        mdr_list = []
        for agent, mdr in MdR4Agents:
            mdr_list.append(mdr)
        mdr_string_num = '-'.join(map("{:02d}".format, mdr_list))
        return mdr_string_num



if __name__ == "__main__":
    main()
