import React from 'react';

import {ComponentStory, ComponentMeta} from '@storybook/react';

import PostSurvey from './PostSurvey';
import Header from "../../Header";

export default {
    title: 'Trials/PostSurvey',
    component: PostSurvey,
} as ComponentMeta<typeof PostSurvey>;

const Template: ComponentStory<typeof PostSurvey> = function (args) {
    return (
        <>
            <Header title={"Final survey"}/>
            <PostSurvey {...args}/>
        </>
    );
};

export const DefaultStory = Template.bind({});

DefaultStory.args = {
    endTrial: (data: any) => console.log(data),
    requiredFields: [true, true, true, true, false]
};